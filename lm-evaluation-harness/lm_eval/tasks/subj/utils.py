
def get_target_from_label(doc):

    # modifies the contents of a single
    # document in our dataset.
    target = "No" if doc['label'] == 0 else "Yes"
    return target


def get_SUL_target_from_label(doc):

    # modifies the contents of a single
    # document in our dataset.
    target = "Bar" if doc['label'] == 0 else "Foo"
    return target