# Robotics And VLA Frameworks

This folder is a survey-style bridge from LLM systems into embodied AI and VLA tooling.

## Files

- `frameworks.ipynb`: main notebook.
- `Robotics.pdf`: companion slides.
- `requirements.txt`: lightweight notebook dependencies.
- `media/`: referenced assets.

## Topics

- Robosuite
- RoboVerse
- MetaSim
- LeRobot
- Practical differences between simulator-heavy and data-centric robotics stacks

## Install

Start with the core notebook dependencies:

```bash
pip install -r requirements.txt
```

Then add whichever framework you want to explore. The notebook includes install snippets, but these ecosystems change often and some require platform-specific setup.

## Run

```bash
jupyter notebook frameworks.ipynb
```

## Validation

The notebook metadata was normalized and the folder-level instructions were cleaned up, but this notebook was not executed end to end locally because the robotics frameworks it references are large, fast-moving, and often simulator-dependent.

## Suggested Use

- Read this as a map of the robotics/VLA landscape.
- Pick one framework and then move to that project’s official install guide.
- Treat the LeRobot and simulator sections as a comparison exercise, not a single turnkey environment.

## References

- [OpenVLA](https://openvla.github.io/)
- [Robosuite](https://robosuite.ai/)
- [RoboVerse](https://roboverse.wiki/)
- [LeRobot](https://huggingface.co/docs/lerobot)
