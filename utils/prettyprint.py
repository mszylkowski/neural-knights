def config_to_markdown_list(config, indent=0):
    """Convert a dictionary into a Markdown formatted list of properties."""
    markdown_list = ""
    for key, value in config.items():
        if type(value) == dict:
            markdown_list += ("  "*indent) + f"- **{key}**:\n"
            markdown_list += config_to_markdown_list(value, indent=indent+1)
        else:
            markdown_list += ("  "*indent) + f"- **{key}**: {value}\n"
    return markdown_list
