from collections import Counter

class ContextSampler:
    def __init__(self, docs, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"
        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs = self.docs.select(fewshot_indices)

    def get_context(self, doc, num_fewshot):
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = (
            num_fewshot + 1
            if (self.config.fewshot_split == self.config.test_split) or (self.config.fewshot_split == self.config.validation_split)
            else num_fewshot
        )

        # draw `n_samples` docs from fewshot_docs
        fewshotex = self.sample(n_samples, doc)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [x for x in fewshotex if x != doc][:num_fewshot]
        #selected_docs = fewshotex[:num_fewshot]
        for x in selected_docs:
            if x == doc:
                print("Warning: doc is in fewshotex")

        labeled_examples = (
            self.fewshot_delimiter.join(
                [
                    # TODO: is separating doc_to_text and doc_to_target by one space always desired?
                    (
                        self.doc_to_text(doc)
                        if (
                            self.config.doc_to_choice is None
                            or isinstance(self.doc_to_text(doc), str)
                        )
                        else self.doc_to_choice(doc)[self.doc_to_text(doc)]
                    )
                    + self.target_delimiter
                    + (
                        str(self.doc_to_target(doc)[0])
                        if isinstance(self.doc_to_target(doc), list)
                        else self.doc_to_target(doc)
                        if (
                            self.config.doc_to_choice is None
                            or isinstance(self.doc_to_target(doc), str)
                        )
                        else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
                    )
                    for doc in selected_docs
                ]
            )
            + self.fewshot_delimiter
        )

        return labeled_examples

    def sample(self, n, doc):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """
        return self.rnd.sample(self.docs, n)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert (
            n <= len(self.docs)
        ), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):

    def sample(self, n, doc) -> None:
        """
        Return approximately class-balanced samples randomly selected from the dataset.
        """
        class_counts = Counter([doc['label'] for doc in self.docs])
        num_classes = len(class_counts)

        # Calculate the base number of samples per class and the remainder
        base_samples_per_class, remainder = divmod(n, num_classes)

        # Allocate additional sample to classes until the remainder is exhausted
        samples_allocation = {cls: base_samples_per_class for cls in class_counts}
        for cls, _ in self.rnd.sample(class_counts.items(), remainder):
            samples_allocation[cls] += 1

        # Collect samples randomly for each class based on the allocation
        samples = []
        for cls, count in samples_allocation.items():
            class_docs = [doc for doc in self.docs if doc['label'] == cls]
            samples.extend(self.rnd.sample(class_docs, min(count, len(class_docs))))

        # Shuffle the final collection of samples
        self.rnd.shuffle(samples)
        return samples
    
class SyntheticSampler(ContextSampler):
    """
    Return samples that are not the same as the query_doc.
    """
    def sample(self, n, query_doc) -> None:

        class_counts = Counter([doc['label'] for doc in self.docs])
        num_classes = len(class_counts)

        # Calculate the base number of samples per class and the remainder
        base_samples_per_class, remainder = divmod(n, num_classes)

        # Allocate additional sample to classes until the remainder is exhausted
        samples_allocation = {cls: base_samples_per_class for cls in class_counts}
        for cls, _ in self.rnd.sample(class_counts.items(), remainder):
            samples_allocation[cls] += 1

        # Collect samples randomly for each class based on the allocation
        samples = []
        for cls, count in samples_allocation.items():
            class_docs = [doc for doc in self.docs if doc['label'] == cls]

            # Split the query to extract item categories
            parts = query_doc['text'].split()
            query_elem1, query_elem2 = parts[0], parts[1]
            
            # Ensure that the query elements are not in the text of the sampled examples
            valid_examples = [doc for doc in class_docs if not (doc['text'].split()[0] == query_elem1 or doc['text'].split()[1] == query_elem2)]
            
            samples.extend(self.rnd.sample(valid_examples, min(count, len(valid_examples))))
        
        self.rnd.shuffle(samples)
        return samples


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
    "balanced": BalancedSampler,
    "synthetic": SyntheticSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}"
        )
