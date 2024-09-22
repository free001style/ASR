import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    sorted_items = sorted(dataset_items, key=lambda item: item['spectrogram'].shape[2], reverse=True)
    batch_size = len(sorted_items)
    max_spectrogram_length = sorted_items[0]['spectrogram'].shape[2]
    max_text_encoded_length = len(max(sorted_items, key=lambda item: len(item['text_encoded']))['text_encoded'])
    freq_length = sorted_items[0]['spectrogram'].shape[1]
    result_batch = {
        'spectrogram': torch.zeros((batch_size, freq_length, max_spectrogram_length)),
        'spectrogram_length': torch.zeros(batch_size),
        'text_encoded': torch.zeros((batch_size, max_text_encoded_length)),
        'text_encoded_length': torch.zeros(batch_size)
    }
    for i in range(batch_size):
        spec = sorted_items[i]['spectrogram'][0]  # F x T
        target = sorted_items[i]['text_encoded']  # L
        result_batch['spectrogram'][i, :, :spec.shape[1]] = spec
        result_batch['spectrogram'][i, :, spec.shape[1]:] = torch.zeros(
            (freq_length, max_spectrogram_length - spec.shape[1]))
        result_batch['spectrogram_length'][i] = spec.shape[1]
        result_batch['text_encoded'][i, :len(target)] = target
        result_batch['text_encoded'][i, len(target):] = torch.zeros(max_text_encoded_length - len(target))
        result_batch['text_encoded_length'][i] = len(target)
    return result_batch
