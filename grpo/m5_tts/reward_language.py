"""Request-scoped ASR/alignment language; scoring weights are unchanged."""
LANGUAGES={'zh':'Chinese','en':'English','de':'German','fr':'French','ja':'Japanese','ko':'Korean'}
def canonical_language(value):
    if not isinstance(value,str) or not value.strip():raise ValueError('Missing reward language')
    value=value.strip()
    for code,name in LANGUAGES.items():
        if value.lower() in (code,name.lower()):return name
    raise ValueError('Unsupported reward language: '+value)
