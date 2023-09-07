# Keras Core Addons

Keras Core Addons is a repository of contributions that conform to well-established API patterns, but implement new 
functionality not available in Keras Core. Keras Core natively supports a large number of operators, layers, metrics, 
losses, and optimizers. However, in a fast moving field like ML, there are many interesting new developments that cannot 
be integrated into core Keras Core (because their broad applicability is not yet clear, or it is mostly used by a 
smaller subset of the community).

Unlike the package this is inspired by (Tensorflow Addons), Keras Core Addons maintains a near similar structure to 
Keras Core, with the `activations`, `layers` and `losses` structure being continued. This is for potential adoption into
Keras Core being as seamless as possible.