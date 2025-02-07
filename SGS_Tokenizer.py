import torch
import numpy as np
import hashlib
import mmh3
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sklearn.metrics.pairwise import cosine_similarity

class ExtendedGPT2Tokenizer(GPT2Tokenizer):
    def __init__(self, vocab_file, merges_file, tensor_0_path='', tensor_1_path='', tensor_2_path='', *args, p=4, int_bit=32, **kwargs):
        super().__init__(vocab_file, merges_file, *args, **kwargs)
        self.p = p
        self.int_bit = int_bit
        self.num_bins = 2 ** p
        self.tensor_0 = torch.load(tensor_0_path) if tensor_0_path else {}
        self.tensor_0_keys = list(self.tensor_0.keys())
        self.tensor_0_values = torch.tensor(list(self.tensor_0.values()))
        self.tensor_1 = torch.load(tensor_1_path) if tensor_1_path else torch.zeros((0, 3), dtype=torch.int64)
        self.tensor_2 = torch.load(tensor_2_path) if tensor_2_path else torch.zeros((0, self.num_bins + 1), dtype=torch.int64)

    def zeros(self, n):
        return min(n + 1, self.int_bit - self.p)
    
    def update_tensors(self, token_ids):
        hashes = self.hash_token_ids(token_ids)
        new_entries = []
        double_values = []
        for token_id, hash_value in zip(token_ids, hashes):
            bin_number, trailing_zeros = self.calculate_bin_and_zeros(hash_value)
            trailing_zeros = self.zeros(trailing_zeros)
            double_value = float(f"{bin_number}.{trailing_zeros}")
            double_values.append(double_value)
            if hash_value not in self.tensor_1[:, 1]:
                new_entries.append(torch.tensor([[token_id, hash_value, double_value]], dtype=torch.float64))
        if new_entries:
            new_tensor_1 = torch.cat(new_entries, dim=0)
            self.tensor_1 = torch.cat((self.tensor_1, new_tensor_1), dim=0)
        return new_tensor_1, double_values

    def tensor_to_hlltensor(self, tensor):
        hlltensor = torch.zeros(self.num_bins, dtype=torch.int64)
        for row in tensor:
            token_id, hash_value, double_value = row
            double_value = float(double_value)  # Ensure double_value is a float
            bin_number, zeros_number = map(int, str(double_value).split('.'))
            zeros_old = hlltensor[bin_number].item()
            hlltensor[bin_number] = self.sequence_to_integer(zeros_old, [zeros_number])
        id, sha1 = self.update_tensor_0(hlltensor)
        self.update_tensor_2(id, hlltensor)
        return id, sha1, hlltensor

    def hlltensor_to_tensor(self, hllset):
        tensor_slice = []
        for bin_number in range(self.num_bins):
            bin_value = hllset[bin_number]
            if bin_value == 0:
                continue
            zeros_array = self.sequence_from_integer(bin_value.item())
            for zeros_number in zeros_array:
                double_value = float(f"{bin_number}.{zeros_number}")
                mask = self.tensor_1[:, 2] == double_value
                matching_rows = self.tensor_1[mask]
                if matching_rows.size(0) > 0:
                    tensor_slice.append(matching_rows)
        return torch.cat(tensor_slice, dim=0) if tensor_slice else torch.zeros((0, 3), dtype=torch.float64)

    def update_tensor_0(self, hlltensor):
        sha1_hash = self.tensor_sha1(hlltensor)
        if sha1_hash not in self.tensor_0:
            self.tensor_0[sha1_hash] = len(self.tensor_0) + 1
        return self.tensor_0[sha1_hash], sha1_hash

    def update_tensor_2(self, id, hlltensor):
        id_tensor = torch.tensor([id], dtype=torch.int64)
        hlltensor_with_id = torch.cat((id_tensor, hlltensor), dim=0).unsqueeze(0)
        self.tensor_2 = torch.cat((self.tensor_2, hlltensor_with_id), dim=0)

    def get_hllset_id(self, hllset):
        hllset_sha1 = hashlib.sha1(hllset.numpy().tobytes()).hexdigest()
        indices = torch.nonzero(torch.eq(self.tensor_0[:, 1], float.fromhex(hllset_sha1)), as_tuple=False)
        return indices[0].item() if indices.numel() > 0 else None

    def tokenize_text(self, text):
        return self.encode(text)

    def search(self, query, threshold):
        token_ids = self.tokenize(query)
        query_hashes = self.hash_token_ids(token_ids)
        query_hllset = np.zeros(self.num_bins, dtype=np.int64)
        for hash_value in query_hashes:
            bin_number, trailing_zeros = self.calculate_bin_and_zeros(hash_value)
            trailing_zeros = self.zeros(trailing_zeros)
            query_hllset[bin_number] = self.sequence_to_integer(query_hllset[bin_number].item(), [trailing_zeros])
        query_hllset = torch.tensor(query_hllset, dtype=torch.int64) if isinstance(query_hllset, np.ndarray) else query_hllset
        query_hllset_with_id = torch.cat((torch.tensor([[0]], dtype=torch.int64), query_hllset.unsqueeze(0)), dim=1)
        similarities = cosine_similarity(query_hllset_with_id.numpy(), self.tensor_2.numpy())
        return [i for i, similarity in enumerate(similarities[0]) if similarity > threshold]

    def get_related_tokens(self, indices):
        relevant_hllsets = self.tensor_2[indices]
        token_ids = []
        for hllset in relevant_hllsets:
            tensor_slice = self.hlltensor_to_tensor(hllset[-self.num_bins:])
            token_ids.extend(row[0].item() for row in tensor_slice)
        return [self.decode([token_id]) for token_id in token_ids]

    def generate_text(self, tokens, num_suggestions=3):
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        input_ids = tokenizer.encode(tokens, return_tensors='pt')
        if input_ids.numel() == 0:
            raise ValueError("Input tokens resulted in an empty tensor.")
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=num_suggestions, no_repeat_ngram_size=2, num_beams=num_suggestions, pad_token_id=tokenizer.eos_token_id)
        return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) for output in outputs]

    def evaluate_generated_texts(self, generated_texts, communities):
        results = []
        community_hllsets = [self.tensor_2[self.tensor_0[hllset_id] - 1][-self.num_bins:] for hllset_id in communities if hllset_id in self.tensor_0]
        for text in generated_texts:
            token_ids = self.tokenize(text)
            text_hashes = self.hash_token_ids(token_ids)
            text_hllset = np.zeros(self.num_bins, dtype=np.int64)
            for hash_value in text_hashes:
                bin_number, trailing_zeros = self.calculate_bin_and_zeros(hash_value)
                trailing_zeros = self.zeros(trailing_zeros)
                text_hllset[bin_number] = max(text_hllset[bin_number], trailing_zeros)
            similarities = cosine_similarity([text_hllset], community_hllsets)
            best_match_idx = np.argmax(similarities)
            results.append((text, communities[best_match_idx], similarities[0][best_match_idx]))
        return results

    def format_generated_texts(self, suggestions):
        return "\nGenerated text suggestions:\n" + "\n\n".join([f'Suggestion {i+1}:\n {suggestion}' for i, suggestion in enumerate(suggestions)])

    def save_tensors(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.tensor_0, os.path.join(directory, 'tensor_0.pt'))
        torch.save(self.tensor_1, os.path.join(directory, 'tensor_1.pt'))
        torch.save(self.tensor_2, os.path.join(directory, 'tensor_2.pt'))

    def sequence_to_integer(self, old_in, new_zeros):
        return old_in | sum(1 << index for index in new_zeros)

    def sequence_from_integer(self, n):
        return [index for index in range(n.bit_length()) if n & (1 << index)]

    def calculate_bin_and_zeros(self, hash_value):
        bin_number = hash_value >> (self.int_bit - self.p)
        trailing_zeros = (hash_value & -hash_value).bit_length() - 1
        return bin_number, trailing_zeros

    def hash_token_id(self, token_id):
        return mmh3.hash64(str(token_id))[0] & 0xffffffff

    def hash_token_ids(self, token_ids):
        return [self.hash_token_id(token_id) for token_id in token_ids]

    def tensor_sha1(self, tensor):
        return hashlib.sha1(tensor.numpy().tobytes()).hexdigest()

    def get_tensor_0(self, key):
        if key not in self.tensor_0:
            raise KeyError(f"Key {key} not found in tensor0")
        index = self.tensor_0_keys.index(key)
        return key, self.tensor_0_values[index].item()

    def print_tensors(self):
        print("tensor_0:", self.tensor_0)
        self.print_tensor_1(self.tensor_1)

    def print_tensor_1(self, tensor=None):
        tensor = tensor if tensor is not None else self.tensor_1
        print("tensor_1:")
        for row in tensor:
            print(f"[{int(row[0])},\t{int(row[1])},\t{float(row[2])}]")

    def get_tensor_values(self, tensor, bin, zeros):
        return tensor[zeros, bin, :].tolist()