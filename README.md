## TODO

- [x] Implement add-run and find-run streamlit helpers
- [x] Implement dataset analysis module
- [x] Implement option to sample element noise from categorical distribution
- [x] Implement optional PCFM projection step
- [x] Implement the ghost atom
- [x] Implement final discrete projection step
- [x] Implement optimal transportation field for elements
- [x] Explore dynamic $\tau$ and PCFM project strength approaches (with regard to $t$ or model confidence over $\theta$)

## BMP Dataset

Model element list: ["Si", "O", "Li", "Al", "Ba", "Be", "Ca", "K", "P", "Ti", "Zn"]
Element distribution: [0.0825, 0.6201, 0.0198, 0.0872, 0.0194, 0.0194, 0.0193, 0.0181, 0.0764, 0.0184, 0.0193]

**With ghost atoms (max_density=0.11):**
Model element list: ["Si", "O", "Li", "Al", "Ba", "Be", "Ca", "K", "P", "Ti", "Zn", "X"]
Element distribution: [0.0555, 0.4168, 0.0133, 0.0586, 0.0130, 0.0130, 0.0130, 0.0122, 0.0513, 0.0124, 0.0130, 0.3279]

**With ghost atoms (max_density=0.10):**
Model element list: ["Si", "O", "Li", "Al", "Ba", "Be", "Ca", "K", "P", "Ti", "Zn", "X"]
Element distribution: [0.0610, 0.4584, 0.0146, 0.0645, 0.0143, 0.0143, 0.0143, 0.0134, 0.0565, 0.0136, 0.0143, 0.2607]
