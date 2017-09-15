# plotly_tools.py - Module providing a set of tools for rendering graphs using Plotly
# 
# Author: Stefan Fuertinger [stefan.fuertinger@esi-frankfurt.de]
# Created: August 10 2017
# Last modified: <2017-09-14 16:51:19>

from __future__ import division
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter, cnames
import plotly.offline as po
import plotly.graph_objs as go
import cmocean
import sys
import h5py
import os

##########################################################################################
def cmap_plt2js(cmap, cmin=0.0, cmax=1.0, Ncols=None):
    """
    Converts Matplotlib colormaps to JavaScript color-scales

    Parameters
    ----------
    cmap : Matplotlib colormap
        A linear segmented colormap (e.g., any member of `plt.cm`). 
    cmin : float
        Lower cutoff value for truncating the input colormap that satisfies `0 <= cmin < 1`. 
    cmax : float
        Upper cutoff value for truncating the input colormap that satisfies `0 < cmax <= 1`. 
    Ncols : int
        Resolution of output color-scale, e.g., if `Ncols = 12`, the output 
        color-scale will contain 12 discrete RGB color increments. 
       
    Returns
    -------
    cscale : list
        List of lists containing `Ncols` float-string pairs encoding value-RGB color 
        assignments  in increasing order starting with `cscale[0] = [0, 'rgb(n_1,m_1,l_1)']`
        up to `cscale[Ncols-1] = [1.0, 'rgb(n_Ncols,m_Ncols,l_Ncols)']`, where 
        `n_1,...,n_Ncols`, `m_1,...,m_Ncols` and `l_1,...,l_Ncols` are integers between 
        0 and 255 (see Examples for details). 
       
    Notes
    -----
    None

    Examples
    --------
    The following command converts and down-samples the jet colormap in the range [0.1, 1] 
    to a JavaScript color-scale with 12 components

    >>> import matplotlib.pyplot as plt
    >>> cscale = cmap_plt2js(plt.cm.jet,cmin=0.1,Ncols=12)
    >>> cscale
    [[0.0, 'rgb(0,0,241)'],
     [0.090909090909090912, 'rgb(0,56,255)'],
     [0.18181818181818182, 'rgb(0,140,255)'],
     [0.27272727272727271, 'rgb(0,224,251)'],
     [0.36363636363636365, 'rgb(64,255,183)'],
     [0.45454545454545459, 'rgb(131,255,115)'],
     [0.54545454545454541, 'rgb(199,255,48)'],
     [0.63636363636363635, 'rgb(255,222,0)'],
     [0.72727272727272729, 'rgb(255,145,0)'],
     [0.81818181818181823, 'rgb(255,67,0)'],
     [0.90909090909090917, 'rgb(218,0,0)'],
     [1.0, 'rgb(128,0,0)']]

    See also
    --------
    None
    """

    # Make sure that our single mandatory input argument makes sense
    if type(cmap).__name__ != 'LinearSegmentedColormap':
        raise TypeError('Input colormap `cmap` has to be a Matplotlib colormap!')

    # Check colormap bounds
    varnames = ["cmin", "cmax"]
    for var in varnames:
        scalarcheck(eval(var),var,bounds=[0.0,1.0])
    if cmin == 1.0:
        raise ValueError("Lower cut-off for colormap must be < 1.0!")
    if cmax == 0.0:
        raise ValueError("Upper cut-off for colormap must be > 0.0!")

    # Check colormap resolution
    if Ncols is None:
        Ncols = cmap.N
    else:
        scalarcheck(Ncols,'Ncols',bounds=[1,np.inf],kind='int')    

    # Create listed colormap and sub-sample/truncate input colormap if wanted
    new_cmap = plt.matplotlib.colors.ListedColormap(cmap(np.linspace(cmin,cmax,Ncols)),
                                                    name="truncated_{}".format(cmap.name))

    # For JavaScript to understand color specifications, we have to first multiply the [0,1]-normed
    # color array by 255 (so that we get a colormap with 256 possible values between 0 and 255) and create
    # a list of value - "rgb(m,n,l)" pairs
    newcols = new_cmap.colors[:,:-1]*255
    cscale = []
    for m, dc in enumerate(np.linspace(0.0, 1.0, Ncols)):
        cscale.append([dc, "rgb("+"".join(str(int(np.round(c)))+"," for c in newcols[m,:])[:-1]+")"])

    return cscale

##########################################################################################
def make_brainsurf(surfname, orientation=True, orientation_lines=True, orientation_labels=True,
                   orientation_lw=2, orientation_lcolor="black", orientation_fsize=18, orientation_fcolor="black",
                   shiny_srf=True, surf_opac = 1.0, surf_color='Gainsboro', view=None):
    """
    Create Plotly graph objects to render brain surfaces

    Parameters
    ----------
    surfname : string
        Name of brain surface dataset in associated HDF5 container (access the container with
        `h5py` and use `h5py.File('brainsurf/brainsurf.h5','r').keys()` to see all available surfaces). 
    orientation : bool
        Flag to control whether axes are rendered to illustrate the employed coordinate system. 
        Use the `orientation_*` keywords for more fine-grained controls. 
    orientation_lines : bool
        Flag to control whether axes illustrating the employed coordinate system are 
        rendered (only relevant if `orientation = True`). To prevent standard Cartesian 
        coordinate axes from being drawn on top of the brain's coordinate axes, use
        the `'scene'` item of the return dictionary in Plotly's `Layout` directive, see 
        Examples below for details. 
    orientation_labels : bool
        Flag to control whether labels ("Anterior", "Posterior", "Inferior", "Superior",
        "Left", "Right") highlighting the employed coordinate system are rendered (only relevant if 
        `orientation = True`). 
    orientation_lw : float
        Line-width of axes illustrating the employed coordinate system (only relevant if 
        both `orientation = True` and `orientation_lines = True`). 
    orientation_lcolor : string
        String to set the color of axes lines. Check `matplotlib.colors.cnames` for supported colors 
        (only relevant if both `orientation = True`  and `orientation_lines = True`). 
    orientation_fsize : int
        Font size of coordinate system labels (only relevant if both `orientation = True` and
        `orientation_labels = True`). 
    orientation_fcolor : string
        String to set the color of axes labels. Check `matplotlib.colors.cnames` for supported colors 
        (only relevant if both `orientation = True`  and `orientation_labels = True`). 
    shiny_srf : bool
        Flag that controls whether the rendered brain surface exhibits good or poor light reflection 
        properties making the surface appear "glossy" (`shiny_srf = True`) or matte. 
    surf_opac : float
        Sets the opacity of the brain surface using a value between 0.0 (fully transparent) 
        and 1.0 (solid). 
    surf_color : str
        String to set the color of the brain surface. Check `matplotlib.colors.cnames` for 
        supported colors 
    view : str
        Camera position. Available options are "Axial", "Sagittal" and "Coronal". If `view` is 
        `None` Plotly's default camera position will be selected. 

    Returns
    -------
    ply_dict : dict
        A dictionary containing at least the item `'brain'` representing the generated Plotly 
        `Mesh3d` object for rendering the brain surface respecting all provided optional arguments. 
        If `orientation = True` the dictionary additionally contains the items `'orientation_lines'`
        (a Plotly `Scatter3d` object representing coordinate system axes) and `'orientation_labels'`
        (a Plotly `Scatter3d` object representing axes labels). If only one of the provided keyword 
        arguments `orientation_lines` or `orientation_labels` was `True`, the output dictionary  
        only contains the respective conform item. 
        If `ply_dict` contains the item `'orientation_lines'` and/or the keyword `view` was not `None`, 
        `ply_dict` further contains the item `'scene'` (a nested dictionary which can be forwarded 
        to Plotly's `Layout` directive to control the visibility of Plotly's default axes and/or
        the initial camera position, see Examples below for details).  
       
    Notes
    -----
    None

    Examples
    --------
    The following command returns a dictionary of Plotly objects to render the 
    `BrainMesh_Ch2withCerebellum` brain as fully opaque glossy surface and sets up 
    the initial camera view point in axial position

    >>> pyt_dict = pyt.make_brainsurf('BrainMesh_Ch2withCerebellum', view='Axial')

    The generated objects can be subsequently used to create a HTML file for rendering
    the surface in a web-browser based on embedded JavaScript code that employs D3.js 
    functionality

    >>> import plotly.offline as po
    >>> import plotly.graph_objs as go
    >>> layout = go.Layout(scene=pyt_dict['scene'])
    >>> fig = go.Figure(data=[pyt_dict['brain'],
                              pyt_dict['orientation_labels'], 
                              pyt_dict['orientation_lines']],
                        layout=layout)
    >>> po.plot(fig, filename='brain.html')

    See also
    --------
    Plotly : A data analytics and visualization tool for generating interactive 
             2D and 3D graphs rendered with D3.js. More information available
             on its `official website <https://plot.ly/>`_
    D3.js : A JavaScript library for visualizing data with HTML, SVG, and CSS.
            More information available on its `official website <https://d3js.org/>`_
    """

    # Start by making sure that we have access to the brain surface container (which is assumed to
    # reside in the sub-directory 'brainsurf' of the local folder) 
    # This beautiful construction creates a string representing the absolute path of this script
    mypath = os.sep.join(os.path.realpath(__file__).split(os.sep)[0:-1])
    h5brain = mypath +os.sep+'brainsurf'+os.sep+'brainsurf.h5'
    try:
        h5brainfile = h5py.File(h5brain,'r')
    except:
        raise IOError("Could not open brain surface HDF5 container "+h5brain+"!")

    # Now check if the provided surface file-name makes sense
    if not isinstance(surfname,(str,unicode)):
        raise TypeError('Brain surface name has to be a string!')
    supported = h5brainfile.keys()
    if surfname not in supported:
        sp_str = str(supported)
        sp_str = sp_str.replace('[','')
        sp_str = sp_str.replace(']','')
        msg = 'Unavailable surface `'+str(surfname)+\
              '`. Available options are: '+sp_str
        raise ValueError(msg)

    # Check surface parameters
    if not isinstance(shiny_srf,bool):
        raise TypeError('Surface lightning atmosphere has to be provided using a binary True/False flag!')
    scalarcheck(surf_opac,'surf_opac',bounds=[0.0,1.0])
    colorcheck(surf_color,'surf_color')
    surf_color = surf_color.lower()

    # Let's see if orientation lines/labels are wanted... 
    for var in [orientation, orientation_lines, orientation_labels]:
        if not isinstance(var,bool):
            raise TypeError('Orientation lines/labels are turned on/off using binary True/False flags!')

    # ... if yes, check orientation line/label parameters
    if orientation:
        varnames = ['orientation_lw', 'orientation_fsize']
        for var in varnames:
            scalarcheck(eval(var),var,bounds=[0,np.inf])
        varnames = ['orientation_lcolor', 'orientation_fcolor']
        for var in varnames:
            colorcheck(eval(var),var)
        orientation_lcolor = orientation_lcolor.lower()
        orientation_fcolor = orientation_fcolor.lower()
    else:
        orientation_lines = False
        orientation_labels = False
        
    # Finally, check `view`
    if view is not None:
        sp_str = "'Axial', 'Sagittal', 'Coronal'"
        if not isinstance(view,(str,unicode)):
            raise TypeError('Brain view must be either `None` or one of '+sp_str+"!")
        view = view[0].upper()+view[1:]
        if sp_str.find(view) < 0:
            raise ValueError("Unavailable view `"+view+"`. Available options are "+sp_str)

    # Now finally start actually doing something
    # Extract vertex coordinates and triangle indices from given brain container
    coords = h5brainfile[surfname]['coord'].value
    tri = h5brainfile[surfname]['tri'].value
    h5brainfile.close()

    # Allocate the return dictionary
    ply_dict = {}

    # ==========================================================================================
    #                        BRAIN SURFACE MESH
    # ==========================================================================================
    # Set surface lightning properties depending on whether we want a shiny "wet" brain or not
    if shiny_srf:
        amb = 0.4
        dif = 0.8
        spec = 1.9
        rough = 0.1
        fres = 0.2
    else:
        amb = 0.5
        dif = 0.6
        spec = 1.9
        rough = 0.99
        fres = 4.99
        
    # Use Plotly's `Mesh3d` to render the surface
    brain = go.Mesh3d(

        # Cartesian coordinates of vertices
        x = coords[:,0],
        y = coords[:,1],
        z = coords[:,2],

        # Corresponding triangle indices
        i = tri[:,0],
        j = tri[:,1],
        k = tri[:,2],

        # Disable mouse interactions and don't let this show up in the legend
        hoverinfo = "none",
        showlegend = False,

        # Set surface opacity and color
        opacity = surf_opac,
        color = "rgb("+"".join(str(int(c))+"," for c in np.array(colorConverter.to_rgb(surf_color))*255)[:-1]+")",

        # Set lighting properties and position
        lighting = dict(
            ambient = amb,
            diffuse = dif,
            specular = spec,
            roughness = rough,
            fresnel = fres
        ),
        lightposition = dict(
            x = -8*1e2,
            y = 0,
            z = 4,
        )
    )
    ply_dict['brain'] = brain

    # ==========================================================================================
    #                        ANTERIOR/POSTERIOR/... ORIENTATION LINES
    # ==========================================================================================
    # If wanted add some "brain-axes"
    if orientation:

        # Extend of orientation lines within Cartesian coordinate system
        xlo, xhi = (-40, 50)
        ylo, yhi = (-110, 50)
        zlo, zhi = (-70, 60)

        # Draw orientation lines by connecting points in 3D space
        if orientation_lines:
    
            lines = go.Scatter3d(

                # Order obviously matters a lot here... 
                x = [xlo, xlo, None, xlo, xlo, None, xlo, xhi],
                y = [ylo, ylo, None, ylo, yhi, None, yhi, yhi],
                z = [zlo, zhi, None, zlo, zlo, None, zlo, zlo],

                mode = 'lines',
                line = dict(
                    width = orientation_lw,
                    color = orientation_lcolor),

                # Disable mouse interactions and don't let this show up in the legend
                hoverinfo = "none",
                showlegend = False
            )
            ply_dict['orientation_lines'] = lines

            # If orientation lines are rendered, don't show Plotly's standard axes
            scene = {}
            for key, value in {"xaxis": "x Axis", "yaxis": "y Axis", "zaxis": "z Axis"}.items():
                scene[key] = dict(visible = False)
            ply_dict['scene'] = scene

        # Give some spatial orientation via textual clues in 3D space
        if orientation_labels:

            pos_info = go.Scatter3d(

                # Again, order matters here: this has to align with the `text` list below
                x = [xlo, xlo, xlo, xlo, xlo, xhi],
                y = [ylo, ylo, ylo-10, yhi+15, yhi, yhi],
                z = [zlo-10, zhi, zlo, zlo, zlo, zlo],

                mode = 'text',
                text = ["Inferior", "Superior", "Posterior", "Anterior", "Left", "Right"],
                # text = ["<i>"+posit+"</i>" for posit in \
                # ["Inferior", "Superior", "Posterior", "Anterior", "Left", "Right"]],
                textfont = dict(
                    size = orientation_fsize,
                    color = orientation_fcolor,
                    ),

                # Disable mouse interactions and don't let this show up in the legend
                hoverinfo = "none",
                showlegend = False
            )
            ply_dict['orientation_labels'] = pos_info

    # ==========================================================================================
    #                        3D CAMERA SETUP
    # ==========================================================================================
    # Set initial camera position (if `view == None` use default camera position)
    if view is not None:

        # import ipdb
        # ipdb.set_trace()

        # Initialize `scene` dict if necessary
        if not orientation_lines:
            scene = {}

        # Set up camera position
        if view == "Axial":
            scene['camera'] = dict(
                eye = dict(
                    x = 0,
                    y = -1e-6,
                    z = 2,
                )
            )
        elif view == "Sagittal":
            scene['camera'] = dict(
                eye = dict(
                    x = -2,
                    y = 0,
                    z = -0.1,
                )
            )
        elif view == "Coronal":
            scene['camera'] = dict(
                eye = dict(
                    x = 0,
                    y = 2,
                    z = -0.1,
                )
            )
        ply_dict['scene'] = scene

    # Throw back the final dictionary
    return ply_dict
    
##########################################################################################
def scalarcheck(val,varname,kind=None,bounds=None):
    """
    Local helper function performing sanity checks on scalars
    """

    if not np.isscalar(val) or not plt.is_numlike(val) or not np.isreal(val).all():
        raise TypeError("Input `"+varname+"` must be a real scalar!")
    if not np.isfinite(val):
        raise TypeError("Input `"+varname+"` must be finite!")

    if kind == 'int':
        if (round(val) != val):
            raise ValueError("Input `"+varname+"` must be an integer!")

    if bounds is not None:
        if val < bounds[0] or val > bounds[1]:
            raise ValueError("Input scalar `"+varname+"` must be between "+str(bounds[0])+" and "+str(bounds[1])+"!")

##########################################################################################
def colorcheck(colname,varname):
    """
    Local helper function performing sanity checks on Matplotlib color strings
    """
    
    if not isinstance(colname,(str,unicode)):
        raise TypeError(varname+' has to be a string!')
    if colname.lower() not in cnames.keys() and colname not in cnames.values():
        msg = "Unsupported color `"+varname+" = "+colname+"`. Check `matplotlib.colors.cnames` for possible choices. "
        raise ValueError(msg)
