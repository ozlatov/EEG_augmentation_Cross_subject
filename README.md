# Physiology-based EEG data augmentation for cross-subject classification

We introduce the MI-EEG data augmentation method that leverages knowledge about neurophysics and the geometry of the head. This method is designed to address variations in EEG data across subjects by introducing physiologically plausible variability. EEG signals are primarily generated by pyramidal neurons in the cortex, which are oriented perpendicular to the cortical surface. However, due to the folding of the cortical surface, the location and orientation of these neurons differ among individuals. This can result in significant differences in how the signals are projected onto the scalp. While the substantial inter-participant differences observed in EEG patterns pose a significant challenge for participant-independent classifiers, we can also exploit this fact to modify data of one participant to mimic data of another~(imaginary) participant.

To improve cross-subject classification accuracy and facilitate robust training of complex models, we propose modifying the EEG data from one subject to simulate data from an imaginary subject in a physiology-informed manner. More specifically, we use a source decomposition method on the given data, localize the dipoles of the current sources, randomly change the location of those dipoles, and recombine the source signals using a forward model with the modified dipoles to produce new natural-like EEG data.
