Introduction
----------------

The coupled snowpack and ice surface energy and mass balance model in Python (COSIPY) solves the energy balance at the surface and is coupled to an adaptive vertical multi-layer subsurface module. In this repository, we available modified COSIPY code for Artesonraju glacier located in the Peruvian Andes, including all input data from september 2016 to august 2017 and scripts used to generate plots of the COSIPY output. The original code is available in https://github.com/cryotools/cosipy and was develoment by Sauter et al. (2020).

In COSIPY_PERU, we included a vertical gradient for the albedo timescale parameter to capture the less frequent occurrence of melting and thus, a slower reduction of albedo, in the higher parts of the glacier (Gurgiser et al., 2013). In addition, to account the effect of light-absorbing particles on the surface albedo, we included a vertical gradient for the ice albedo due to in Cordillera Blanca glaciers the lower parts have a higher content of LAPs than the higher parts and thus, the surfaces of the lower parts has a lower albedo than the higher surfaces (Schmitt et al., 2015).


Documentation
-------------

The documentation, input and output variables for COSIPY is available at the following link from code original:
https://cosipy.readthedocs.io/en/latest/

Communication and Support
-------------------------

We are using the groupware slack for communication (inform about new releases, bugs, features, ..) and support:
https://cosipy.slack.com

References
-----

Gurgiser, W., Marzeion, B., Nicholson, L., Ortner, M., and Kaser, G.: Modeling energy and mass balance of Shallap Glacier, Peru, The Cryosphere, 7, 1787–1802, https://doi.org/10.5194/tc-7-1787-2013, 2013.

Sauter, T., Arndt, A., and Schneider, C.: COSIPY v1.3 – an open-source coupled snowpack and ice surface energy and mass balance model, Geosci. Model Dev., 13, 5645–5662, https://doi.org/10.5194/gmd-13-5645-2020, 2020.

Schmitt, C. G., All, J. D., Schwarz, J. P., Arnott, W. P., Cole, R. J., Lapham, E., and Celestian, A.: Measurements of light-absorbing particles on the glaciers in the Cordillera Blanca, Peru, The Cryosphere, 9, 331–340, https://doi.org/10.5194/tc-9-331-2015, 2015.


