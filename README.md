# In-Silico Genome Architecture Mapping
Easy to use, wicked fast simulation of the <a href="https://www.nature.com/articles/nature21411"> Genome Architecture Mapping</a> assay [1]. 
<div align="center">
  <img src="https://github.com/lrburack/In-Silico-GAM/assets/121359508/f979f302-6257-401b-adff-af014bc27333">
</div>

<h2> Explore a range of experimental protocols, no hassle </h2>
In-Silico GAM can be quickly configured to account for 
<br></br>

- homologous chromosomes
- finite sequencing efficiency
- alternative protocols for sampling nuclear profiles
- multiplexing

Nuclear profiles are taken, then multiplexing, finite efficiency, and homology, are applied in post at lightspeed. Skeptical about uniform sampling? Me too. You can easily define your own protocol for slice sampling. This is all packaged in the GAM configuration class which contains everything necessary to run the experiment and process the results.

<h2> How to use... </h2>
Download the stuff. Create an instance of the GAM class, which takes six optional parameters: slice width, multiplexing, detection probability, pickslice function, homolog map. Then call the 'run' function with the ensemble of structures you want to use, and the number of nuclear profiles to take per structure.
<br></br>
utilities.py contains some useful functions for getting your structures in the correct format.  

example.ipynb contains a super simple demonstration of how use the class, and also makes some pretty pictures.  

For documentation see the wiki. 

<h2> Acknowledgements </h2>
Produced with mentorship of Bernardo Zubillaga Herrera PhD and other very smart people at the Di Pierro Lab. Funded by Northeastern University's PEAK Ascent Award.
<br></br>
<div align="center">
  <img src="https://github.com/lrburack/In-Silico-GAM/assets/121359508/dab21206-5f0e-4717-a1f1-27153e803637">
</div>
