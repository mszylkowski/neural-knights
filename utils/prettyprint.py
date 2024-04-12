def config_to_markdown(config_path, config):
    """Convert a dictionary into a Markdown formatted list of properties."""
    markdown_list = f"###Config path: {config_path}\n"
    markdown_list += config_to_markdown_list(config)
    return markdown_list

def config_to_markdown_list(config, indent=0):
    """Recursive traversal of dict. Returns markdown str."""
    markdown_list = ""
    for key, value in config.items():
        if type(value) == dict:
            markdown_list += ("  "*indent) + f"- **{key}**:\n"
            markdown_list += config_to_markdown_list(value, indent=indent+1)
        else:
            markdown_list += ("  "*indent) + f"- **{key}**: {value}\n"
    return markdown_list
