# Generative Adversarial Networks for Time Series Synthetic Data Generation to Monitor Energy Consumption of CNC Machines

## Project Description
This project focuses on the generation of synthetic data to model and predict the performance and energy requirements of CNC (Computer Numerical Control) machines. By leveraging the capabilities of Generative Adversarial Networks (GANs), the project aims to simulate realistic data that mirrors the operational patterns of CNC machinery, enabling more robust predictive analytics and enhancing decision-making for future machine maintenance and energy management.

**Three distinct GAN architectures were implemented for synthetic data generation:**

**[TimeGAN](https://github.com/jsyoon0823/TimeGAN)**: A GAN variant designed for time-series data, TimeGAN employs a Recurrent Neural Network (RNN)-based architecture, making it well-suited for modeling sequential dependencies and patterns in time-based CNC machine data. This model captures temporal dependencies, providing realistic synthetic datasets that resemble actual CNC machine operation sequences.

**TransformerGAN**: Leveraging the powerful Transformer architecture, this GAN is adept at capturing long-range dependencies and intricate data patterns. TransformerGAN provides high-quality synthetic data, especially where complex interactions and dependencies are present, enabling better insights into the nuanced performance characteristics of CNC machines.

**WGAN (Wasserstein GAN):** Known for improved training stability, the WGAN model ensures high-quality synthetic data generation by minimizing the Wasserstein distance, leading to more accurate modeling of CNC machine performance metrics and power consumption data.
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://example.com/build-status)


## Features

- **Transformer-based GAN** for realistic time-series data generation.

- **Metrics for evaluation**:
  - **Descriptive**: Comparison of statistical properties between real and synthetic data. The goal is to compute  how well the discriminator can distinguish between real and synthetic data. The discriminative score is defined as the absolute difference between the classification accuracy and 0.5. An accuracy close to 0.5 indicates that the discriminator cannot distinguish real from synthetic data, which is ideal in a well-trained GAN.
  - **Predictive**: Assessment of the ability to predict future machine performance using synthetic data. The predictive score is a critical metric for assessing and validating the effectiveness of generative models, particularly in time-series contexts. It helps ensure that the synthetic data produced is reliable and representative of the original data,
  - **Frechet Inception Score (FIS)**: For assessing the quality of the generated data. It measure the similarity between the real and synthetic datsets.

- **Visualizations**:
  - **PCA (Principal Component Analysis)**
  - **t-SNE (t-distributed Stochastic Neighbor Embedding)**

- **Model Performance Tracking**: We are utilising the weights and biases library to track the performance of our model. With this, we are getting a dashboard with all the evaluation metrics and other performance measures.

## Structure:

The code is structured into following main components:

model: Contains the transformer-based GAN architecture.
metrics: Implements the evaluation metrics (descriptive, predictive, and Frechet Inception Score).
main: This script serves as the entry point for the project, orchestrating the entire pipeline for synthetic data generation, evaluation, and visualization.


To run the project locally, follow these steps:

1. Clone the repository:
   
   git clone <repository-url>
   cd <project-folder>

2. Modify the configuration files if needed (for custom datasets, hyperparameters, etc.).
