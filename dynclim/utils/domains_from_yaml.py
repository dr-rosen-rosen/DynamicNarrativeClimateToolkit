"""YAML loading utilities for safety domain definitions."""

import yaml
from typing import Dict, List

def loadDomainsFromYaml(yaml_file_path: str) -> Dict[str, List[str]]:
    """
    Load safety domain definitions from YAML file and convert to analysis-ready dict.
    
    Args:
        yaml_file_path: Path to the YAML file containing domain definitions
        
    Returns:
        Dictionary with domain abbreviations as keys and lists of items as values
        Example: {"LSC": ["item1", "item2", ...], "SC": ["item1", "item2", ...], ...}
    """
    with open(yaml_file_path, 'r', encoding='utf-8') as file:
        yaml_data = yaml.safe_load(file)
    
    # Extract domains from the YAML structure
    domains = yaml_data['domains']
    
    # Convert to the desired format: {domain_abbreviation: [list_of_items]}
    domain_definitions = {}
    for domain_abbrev, domain_info in domains.items():
        domain_definitions[domain_abbrev] = domain_info['items']
    
    return domain_definitions

# from typing import Dict, List, Union
# import yaml

# def loadDomainsFromYaml(yaml_file_path: str) -> Dict[str, List[Dict[str, Union[str, bool]]]]:
#     """
#     Load safety domain definitions from YAML file and convert to analysis-ready dict.
#     Supports both legacy (string-only items) and new (dict with id/text/reverse) formats.
    
#     Args:
#         yaml_file_path: Path to the YAML file containing domain definitions
        
#     Returns:
#         Dictionary with domain abbreviations as keys and lists of item dicts as values.
#         Example:
#             {
#                 "TW": [
#                     {"id": "TW1", "text": "In this unit...", "reverse": False},
#                     ...
#                 ],
#                 ...
#             }
#     """
#     with open(yaml_file_path, 'r', encoding='utf-8') as file:
#         yaml_data = yaml.safe_load(file)

#     domains = yaml_data.get('domains', {})

#     domain_definitions = {}
#     for domain_abbrev, domain_info in domains.items():
#         items = []
#         for i, item in enumerate(domain_info.get('items', []), start=1):
#             if isinstance(item, str):
#                 # Legacy format: only text
#                 items.append({
#                     'id': f"{domain_abbrev}{i}",
#                     'text': item,
#                     'reverse': False
#                 })
#             elif isinstance(item, dict):
#                 # New structured format
#                 items.append({
#                     'id': item.get('id', f"{domain_abbrev}{i}"),
#                     'text': item.get('text', ''),
#                     'reverse': item.get('reverse', False)
#                 })
#         domain_definitions[domain_abbrev] = items

#     return domain_definitions