Suppressing Streak Artifacts Generated by the Interference of Imaging and Therapy Fields: Initial Findings Using a Hybrid U-Net and Diffusion Model

Background, Motivation, and Objective: Image guidance through B-mode ultrasound is vital in guiding therapeutic ultrasound interventions. Images acquired when the therapy transducer is firing have streak artifacts caused by the interference of therapy and imaging fields. These artifacts degrade image quality and distract from the therapeutic target. This study aimed to evaluate a deep learning-based approach for streak removal. 

Statement of Contribution/Methods: We combined U-Net architecture with High-Resolution Image Synthesis with Latent Diffusion models to remove streak artifacts from HIFU images. We created a synthetic dataset to train the model by artificially introducing streaks into 20 streak-free ultrasound images and obtained 1638 images. These streaks were varied in spacing, blend factors, intensity, and lengths to closely emulate real-world scenarios. This method generated a sufficiently diverse training dataset without labeled real-world examples. The trained model was then applied to remove artifacts from 154 images acquired while insonating a wall-less polyvinyl alcohol phantom perfused with perfluorohexane droplets at 5109  droplets/ml undergoing phase transition. The insonation frequency, peak negative pressure, pulse length, and pulse repetition period were 2 MHz, 7.4 MPa, 100 cycles, and 10 ms, respectively. The B-Mode images were collected using a Vantage 128 system equipped with a L-11-5v linear array (center frequency of 7.6 MHz).

Results and Discussions: Our results demonstrate the ability of our model to remove streak artifacts while preserving contrast from microbubbles. We see an increase in the Signal-to-Noise ratio by 6.0 ± 3.4 dB (mean ± standard deviation, n=154 frames) after the processing using the Hybrid U-Net and Diffusion Model. The model's rapid convergence underscores its potential for real-time clinical application.

Conclusion: Integrating U-Net with diffusion models provides a potential solution for streak artifact removal in therapeutic ultrasound imaging. Future work will focus on validating the model with more extensive and diverse datasets and reducing inference times.


