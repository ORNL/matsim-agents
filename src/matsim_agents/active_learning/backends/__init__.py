"""DFT backend implementations (VASP, Quantum ESPRESSO).

Both backends conform to :class:`matsim_agents.active_learning.dft_backend.DFTBackend`.
Importing this package does not import either backend's heavy dependencies;
the loop instantiates only the backend selected by ``dft.backend``.
"""
