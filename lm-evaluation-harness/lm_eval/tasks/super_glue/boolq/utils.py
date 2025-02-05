
def get_target_from_label(doc):

    # modifies the contents of a single
    # document in our dataset.
    target = "yes" if doc['label'] == 1 else "no"
    return target
