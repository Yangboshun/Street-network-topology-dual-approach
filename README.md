# Street network topology modeling based on every-best-fit--dual approach
Graduate project, LSGI Department, Hong Kong Polytechnic University

This project supports my own report, it was interesting to explore the underlying characteristics of the street network. I modeled dual graphs of the street networks using Python-based correlation functions from Openstreetmap. This are codes that applies to regions all over the world., while supporting continuity negotiation with Every best-fit strategy.

The picture shows the building principle of dual graph, quote from: https://doi.org/10.1103/PhysRevLett.94.028701

<img width="421" alt="image" src="https://github.com/Yangboshun/Street-network-topology-modeling-based-on-every-best-fit--dual-approach/assets/158756807/e34d82ee-a759-4fb8-897e-1d136e3d05aa">

Mainly import OSMnx, NetworkX, Cions and other packages

Here are some visual examples for reference：

Original network topology:

<img width="331" alt="image" src="https://github.com/Yangboshun/Street-network-topology-modeling-based-on-every-best-fit--dual-approach/assets/158756807/ff6ef95f-50df-4e76-9c2e-fa7a3e8a603b">

Dual graph of the original network topology:

<img width="378" alt="image" src="https://github.com/Yangboshun/Street-network-topology-modeling-based-on-every-best-fit--dual-approach/assets/158756807/c7a721a8-351e-4dfe-88a2-7fdbc81574de">

Every best-fit Strategy Merged streets:

<img width="331" alt="image" src="https://github.com/Yangboshun/Street-network-topology-modeling-based-on-every-best-fit--dual-approach/assets/158756807/1738395a-1004-44c6-8dad-dc2a4350bcf6">

Dual graph：

<img width="378" alt="image" src="https://github.com/Yangboshun/Street-network-topology-modeling-based-on-every-best-fit--dual-approach/assets/158756807/4a76f065-e450-45f9-b283-678ff377dfdb">
