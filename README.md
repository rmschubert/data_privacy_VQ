# Vector Quantization and Data Privacy
This repository is contains the code for the experiments of the submission *"About Vector Quantization and its Privacy in Federated Learning"* by Ronny Schubert and Thomas Villmann (2024).

## How to run the Experiments
We setup the virtual environment via [uv](https://github.com/astral-sh/uv). Follow these steps to recreate the environment

```bash
uv venv
source .venv/bin/activate
uv pip sync requirements.txt
```

Change your directory to the *src* directory and in the active environment run
```bash
python -m federate_attack
```
The results are stored in *src/results* yielding the plots found in the above mentioned submission as well as a *.pkl* file containing all numerical results together with the images and stored prototypes.
