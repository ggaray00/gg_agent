"""Built-in provider profiles, one vendor per module.

Each module registers its profile(s) at import; ``providers/__init__.py``
imports every non-underscore module here. To add a provider, drop a file in this
package (or in ``$GG_HOME/plugins/model-providers/``) that calls
``register_provider``. Nothing else needs to change.

Mirrors hermes-agent: plugins/model-providers/
"""
