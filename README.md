# Coulomb Simulation
Simulating charge movement using Coulomb's Law.

Reference:
[[https://en.wikipedia.org/wiki/Coulomb's_law]]

Example:
Movement of electrons in an electric field.
[![electrons in E-field](https://thumbnails.odycdn.com/optimize/s:1280:720/quality:85/plain/https://thumbs.odycdn.com/77efa1e54c033b53a58140f559a3551e.webp)](https://odysee.com/@mielke.chemie:e/e_field:5)



The purpose of the python code in this repo is to generate a visual display of moving charged particles for middle school lessons (in Dutch language, hence the labels).
I chose an object-oriented approach to getting charged particles (electrons, protons) to force each other into movement.
There is some cheating involved: the dielectric constant is scaled (to make movement visible), time steps are discrete, spatial dimensions are arbitrary.

Due to these flaws and limitations, it takes some parameter tweaking to get a meaningful animation.
The most trouble was caused by particles in close proximity: due to the "1/r²" part in the equation, small distances mean huge accelerations. 
Though this is plausible in principle, the discrete time steps used herein often led to huge jumps of the particles. 
To avoid this, I set small time stamps and added a crude implementation of elastic collision. 
These settings have to be tamed further by a deceleration factor, and some randomness sprinkled on top.
Surprising how difficult it is (i.e. how much crunching is necessary) to see such a simple law in action. However, results are visually acceptable and serve the illustration purpose.

Please take this as "Little science, much education." (Will have to see whether this is a good approach.)



Comments and suggestions are welcome!
