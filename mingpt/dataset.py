import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed


class CopyDataset(Dataset):
    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
        self.sep_id = 0
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits + 1 # +1 for sep
    
    def get_block_size(self):
        # length of the sequence that will feed into transformer
        # full sequence input + sep + all but last output
        return self.length * 2

    def __getitem__(self, idx):
        while True:
            inp = torch.randint(1, self.num_digits + 1, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5: 
                # half of the time boost samples with repeats
                if inp.unique().nelement() > self.length // 2:
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # concatenate the problem specification and the solution
        cat = torch.cat((inp, torch.tensor([self.sep_id], inp, dtype=torch.long)), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length] = -1
        return x, y


class ParallelCopyDataset(Dataset):
    """ 
    Parallel Copy Task: copy an input sequence using multiple decoding tokens per forward pass
    Produces input, labels, sequence positions and attention mask
    
    ** Example **
    seq_len = 7, num_threads = 3
    input tokens:   t0 t1 t2 t3 t4 t5 t6
    special tokens: s0 s1 (and sep)
    
    input:        t0 t1 t2 t3 t4 t5 t6  sep s0 s1 t0 t1 t2 s0 s1 t3 t4 t5
    labels:       -1 -1 -1 -1 -1 -1 -1   t0 t1 t2 t1 t2 t3 t4 t5 t4 t5 t6
    loss mask:    0  0  0  0  0  0  0    1  1  1  1  1  1  1  1  1  1  1  
    attn mask:    1  1  1  1  1  1  1    1  0  0  1  1  1  0  0  1  1  1  
    seq pos:      0  1  2  3  4  5  6    7  8  9  8  9  10 11 12 11 12 13 
    thread id:    .  .  .  .  .  .  .    0  1  2  .  .  0  1  2  .  .  0  
    
    ** Explanation **
    The input has special tokens s0 and s1 which indicate that at the time of prediction,
    the token identity is unknown, and that the last known token is i + 1 tokens ago for s0, s1, etc.
    
    There is an interleaving of the input tokens with the special tokens, which is why we can 
    still parallelize the decoding. In particular, the special tokens are never attended to 
    other than potentially from the other special tokens at that decoding group step.
    
    All tokens have to attend to the full true sequence so far, so we also have to 
    have the normal tokens as inputs after the special tokens. 
    These are what are attended to in future decoding steps instead of the special tokens. 
    
    The thread id indicates which thread is responsible for that decoding step.
    Given the depth of the network and the structure + meaning of the input, 
    this setup allows the model to copy the input sequence correctly in parallel.
    """

    def __init__(self, split, length=6, num_digits=3, extra_threads=4, thread_mask_type='causal'):
        assert split in {'train', 'test'}
        assert extra_threads > 0
        self.split = split
        self.length = length
        self.num_digits = num_digits
        self.extra_threads = extra_threads
        self.num_threads = self.extra_threads + 1
        self.thread_mask_type = thread_mask_type
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits + self.extra_threads + 1 # +1 for sep
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2

    def _get_local_mask(self, length):
        if self.thread_mask_type == 'causal':
            return torch.tril(torch.ones(length, length))
        elif self.thread_mask_type == 'non_causal':
            return torch.ones(length, length)
        elif self.thread_mask_type == 'independent':
            return torch.eye(length)
        
    def _get_output_seq_segments(self, input_seq):
        """
        output is the later part of the input sequence
        In the example it's sep s0 s1 t0 t1 t2 s0 s1 t3 t4 t5
        segments: [sep, s0, s1], [t0, t1], [t2, s0, s1], [t3, t4], [t5]
        """
        sep_id = self.num_digits
        thread_tokens = torch.arange(sep_id + 1, sep_id + 1 + self.extra_threads, dtype=torch.long)
        output_segments = []
        pos = 0
        while pos < self.length:
            inp_tokens = input_seq[pos:pos + self.num_threads]
            if pos == 0:
                inp_tokens = torch.cat((torch.tensor([sep_id], dtype=torch.long), inp_tokens))
            output_segments.extend([thread_tokens, inp_tokens])
            pos += self.num_threads
        return output_segments

    # Helper function to generate full_seq
    def _generate_full_seq(self, input_seq):
        """ 
        input_seq: t0 t1 t2 t3 t4 t5 t6
        output:    sep s0 s1 t0 t1 t2 s0 s1 t3 t4 t5
        full_seq = input_seq + output
        """
        output_segments = self._get_output_seq_segments(input_seq)
        output = torch.cat(output_segments)
        return torch.cat((input_seq, output))

    # Helper function to generate full_seq_pos
    def _generate_full_seq_pos(self):
        """
        input_seq_pos: 0  1  2  3  4  5  6
        output_seq_pos: 7  8  9  8  9  10 11 12 11 12 13
        output_segments_pos: [7, 8, 9], [8, 9], [10, 11, 12], [11, 12], [13]
        """
        input_seq_pos = torch.arange(self.length, dtype=torch.long)
        output_segments_pos = []
        pos = self.length
        ptr = 0
        while ptr < self.length:
            thread_pos = torch.arange(pos, pos + self.num_threads, dtype=torch.long)
            input_pos = torch.arange(pos + 1, pos + self.num_threads, dtype=torch.long)
            output_segments_pos.extend([thread_pos, input_pos])
            pos += self.num_threads
            ptr += self.num_threads
        output_pos = torch.cat(output_segments_pos)
        return torch.cat((input_seq_pos, output_pos))

    # Helper function to generate full_label_seq
    def _generate_full_label_seq(self, input_seq):
        """ 
        label_input_seq: -1 -1 -1 -1 -1 -1 -1
        label_output_seq: t0 t1 t2 t1 t2 t3 t4 t5 t4 t5 t6
        label_output_segments: [t0, t1, t2], [t1, t2], [t3, t4, t5], [t4, t5]
        """
        label_input_seq = torch.full_like(input_seq, fill_value=-1)
        label_output_segments = []
        ptr = 0
        while ptr < self.length:
            thread_labels = input_seq[ptr:ptr + self.num_threads]
            input_labels = input_seq[ptr + 1:ptr + self.num_threads]
            label_output_segments.extend([thread_labels, input_labels])
            ptr += self.num_threads
        label_output_seq = torch.cat(label_output_segments)
        return torch.cat((label_input_seq, label_output_seq))

    def _generate_thread_mask(self, input_seq):
        """ 
        create a mask that allows each thread to attend to the correct tokens
        - start with causal mask for all tokens
        - then ensure that no token attends backward to special tokens outside of the local decoding group
        - for the local decoding groups select the mask type based on the thread_mask_type parameter
        """
        sep_id = self.num_digits
        thread_tokens = torch.arange(sep_id + 1, sep_id + 1 + self.extra_threads, dtype=torch.long)
        output_segments = self._get_output_seq_segments(input_seq)
        full_seq_length = input_seq.size(0) + torch.cat(output_segments).size(0)
        start_mask = torch.tril(torch.ones(full_seq_length, full_seq_length))
        pos = 0
        for segment in output_segments:
            segment_length = segment.size(0)
            if thread_tokens[0] in segment:
                start_mask[pos + segment_length:, pos:pos + segment_length] = 0
                local_mask = self._get_local_mask(segment_length)
                try:
                    start_mask[pos - 1:pos + segment_length, pos - 1:pos + segment_length] = local_mask
                except:
                    import ipdb; ipdb.set_trace()
            pos += segment_length
        return start_mask

    def __getitem__(self, idx):
        while True:
            # generate some random integers
            input_seq = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            if torch.rand(1).item() < 0.5: 
                # half of the time boost samples with repeats
                if input_seq.unique().nelement() > self.length // 2:
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(input_seq.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok

        full_seq = self._generate_full_seq(input_seq)
        full_seq_pos = self._generate_full_seq_pos()
        full_label_seq = self._generate_full_label_seq(input_seq)
        attn_mask = self._generate_thread_mask(input_seq)

        return (full_seq, full_seq_pos, attn_mask), full_label_seq