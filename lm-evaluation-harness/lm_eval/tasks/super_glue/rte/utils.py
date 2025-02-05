
def get_target_from_label(doc):

    # modifies the contents of a single
    # document in our dataset.
    target = "Yes" if doc['label'] == 0 else "No"
    return target
