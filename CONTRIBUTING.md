# Contributing to Keras Core Addons

Interested in contributing to Keras Core Addons? We appreciate all kinds
of help and are working to make this guide as comprehensive as possible.
Please [let us know](https://github.com/terminalai/kca/issues) if
you think of something we could do to help lower the barrier to
contributing.

## Pull Requests

We gladly welcome [pull requests](
https://help.github.com/articles/about-pull-requests/).

Have you ever done a pull request with GitHub? 
If not we recommend you to read 
[this guide](https://github.com/gabrieldemarmiesse/getting_started_open_source) 
to get your started.

Before making any changes, we recommend opening an issue (if it
doesn't already exist) and discussing your proposed changes. This will
let us give you advice on the proposed changes. If the changes are
minor, then feel free to make them without discussion.

All submissions, including submissions by project members, require
review.


## Requirements for New Contributions to the Repository

**All new components/features to Addons need to first be submitted as a feature 
request issue.**

The `kca` repository contains additional functionality fitting the following criteria:

* The functionality is not otherwise available in Keras Core (if it is added, we will remove features)
* Addons have to be compatible with the current version of Keras Core
* The addon conforms to the code and documentation standards
* The addon is impactful to the community (e.g. an implementation used
 in widely cited paper)
 * Lastly, the functionality conforms to the contribution guidelines of
 its subpackage.

Suggested guidelines for new feature requests:

* The feature contains an official reference implementation.
* Should be able to reproduce the same results in a published paper.
* The academic paper exceeds 50 citations.

**Note: New contributions often require team-members to read a research
paper and understand how it fits into the Keras Core community. This
process can take longer than typical commit reviews so please bare with
us!**


## Commit Guidelines

Any and all commits must follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) 
format, regardless of branch. If this is not abided by, commits will be force-pushed to confirm with the
guidelines established.
