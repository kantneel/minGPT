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
    The tra
    
    ** Example **
    seq_len = 7, num_threads = 3
    input tokens:   t0 t1 t2 t3 t4 t5 t6
    special tokens: s0 s1 (and sep)
    
    input:        t0 t1 t2 t3 t4 t5 t6    sep s0 s1 t0 t1 t2 s0 s1 t3 t4 t5
    output:       t1 t2 t3 t4 t5 t6 sep    t0 t1 t2 t1 t2 t3 t4 t5 t4 t5 t6
    loss mask:    0  0  0  0  0  0  0      1  1  1  1  1  1  1  1  1  1  1  
    attn mask:    1  1  1  1  1  1  1      1  0  0  1  1  1  0  0  1  1  1  
    seq pos:      0  1  2  3  4  5  6      7  8  9  8  9  10 11 12 11 12 13 
    thread id:    .  .  .  .  .  .  .      0  1  2  .  .  0  1  2  .  .  0  
    
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
        return self.length * 2 - 1

    def get_local_mask(self, length):
        if self.thread_mask_type == 'causal':
            return torch.tril(torch.ones(length, length))
        elif self.thread_mask_type == 'non_causal':
            return torch.ones(length, length)
        elif self.thread_mask_type == 'independent':
            return torch.eye(length)

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

        def get_output_seq_segments(input_seq):
            sep_id = self.num_digits
            thread_tokens = torch.arange(sep_id + 1, sep_id + 1 + self.extra_threads, dtype=torch.long)
            output_segments = [torch.tensor([sep_id], dtype=torch.long)]
            pos = 0
            while pos < self.length:
                inp_tokens = input_seq[pos:pos + self.num_threads]
                output_segments.extend([thread_tokens, inp_tokens])
                pos += self.num_threads
            return output_segments
        
        # Helper function to generate full_seq
        def generate_full_seq(input_seq):
            output_segments = get_output_seq_segments(input_seq)
            output = torch.cat(output_segments)
            return torch.cat((input_seq, output))

        # Helper function to generate full_seq_pos
        def generate_full_seq_pos():
            input_seq_pos = torch.arange(self.length, dtype=torch.long)
            output_segments_pos = [torch.tensor([self.length], dtype=torch.long)]
            pos = self.length + 1
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
        def generate_full_label_seq(input_seq):
            sep_id = self.num_digits
            label_input_seq = torch.cat((input_seq[1:], torch.tensor([sep_id], dtype=torch.long)))
            label_segments = []
            ptr = 0
            while ptr < self.length:
                thread_labels = input_seq[ptr:ptr + self.num_threads]
                input_labels = input_seq[ptr + 1:ptr + self.num_threads]
                label_segments.extend([thread_labels, input_labels])
                ptr += self.num_threads
            label_seq = torch.cat(label_segments)
            return torch.cat((label_input_seq, label_seq))
        
        def generate_thread_mask(input_seq):
            sep_id = self.num_digits
            thread_tokens = torch.arange(sep_id + 1, sep_id + 1 + self.extra_threads, dtype=torch.long)
            output_segments = get_output_seq_segments(input_seq)
            full_seq_length = input_seq.size(0) + torch.cat(output_segments).size(0)
            start_mask = torch.tril(torch.ones(full_seq_length, full_seq_length))
            pos = 0
            for segment in output_segments:
                segment_length = segment.size(0)
                if thread_tokens[0] in segment:
                    start_mask[pos + segment_length:, pos:pos + segment_length] = 0
                    local_mask = self.get_local_mask(segment_length)
                    start_mask[pos - 1:pos + segment_length, pos - 1:pos + segment_length] = local_mask
                pos += segment_length
            return start_mask
            
        full_seq = generate_full_seq(input_seq)
        full_seq_pos = generate_full_seq_pos()
        full_label_seq = generate_full_label_seq(input_seq)
        attn_mask = generate_thread_mask(input_seq)

        x = full_seq[:-1].clone()
        x_pos = full_seq_pos[:-1].clone()
        y = full_label_seq[1:].clone()
        
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length] = -1
        return (x, x_pos, attn_mask), y