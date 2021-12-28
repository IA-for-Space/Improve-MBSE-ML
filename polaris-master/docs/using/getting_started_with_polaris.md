---
html_meta:
  "description lang=en": "All you need to know to install Polaris and get started with deep learning for spacecraft telemetry and anomaly analysis"
  "keywords": "Polaris, getting started, how-to, guide, installation, spacecraft, operations, telemetry, anomaly, data, machine learning, deep learning, graph, analysis"
  "property=og:locale": "en_US"
  "property=og:title": "Polaris, machine learning for spacecraft operations"
---

Want to get started with Polaris?  Great!  This page will help you do just that.

# Quick install

If you know what you're doing and have a python environment ready (Python version <= 3.9, latest version of `pip`), then you can run this to install the latest polaris release:

```
pip install polaris-ml
```

For users who want to run the last features from our master branch please use:
```
pip install git+https://gitlab.com/librespacefoundation/polaris/polaris
```

# Prerequisites

- (mandatory) A Python 3.x environment
    - Polaris is known to work with python 3.7, 3.8 and 3.9.
    - Python 2 is [not supported](https://www.python.org/doc/sunset-python-2/).

- A Linux or Unix-like operating environment
    - The core team is mainly working on Linux, we're open to whatever you work on.
    - A few contributors use OS X and it should work just fine.
    - Many have tried on Windows and ended up using emulators or virtual machines, give it a try and [let us know how it works for you](https://app.element.io/#/room/#polaris:matrix.org).

- We strongly recommend using a python virtual environment that does not impact your system python environment (by using virtualenv or conda)
    - In next section, we remind shortly how to create your own python virtual environment.
    - [This guide should help you out with virtualenv](https://docs.python-guide.org/dev/virtualenvs/).
    - [This guide should help you out with conda, if you use Anaconda already](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

- You'll need to be comfortable with the command line interface.

- You'll need a good network connection, so you can download telemetry quickly.

# Installing Polaris

(how-to-virtual-environment)=
## Creating a python virtual environment

Create and activate a virtual environment using `virtualenv`:
```{code-block} bash
$ python -m venv .polarisenv
$ source .polarisenv/bin/activate
(.polarisenv) $ pip install --upgrade pip
```
Create and activate a virtual environment using `conda`:
```{code-block} bash
$ conda create -n polarisenv python=3.8
$ source activate polarisenv
(polarisenv) $ pip install --upgrade pip
```

```{admonition} Good to know
:class: info
You will need either virtualenv (system install or pip install virtualenv) or conda (via Anaconda or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)) to be installed. With conda, you can create a python environment from any python version, without the need to pre-install that version.
```

```{admonition} Whitespaces in a virtual environment's path
:class: warning
Various things will break if there is any whitespace in the full path to your virtualenv.  Here are some example of paths that will not work:

* `/home/jdoe/My Development Stuff/polaris/.venv`
* `/mnt/C/Jane Doe/dev/polaris/.venv`

Instead, use paths without spaces in them:

* `/home/jdoe/my_development_stuff/polaris/.venv`
* `/mnt/c/dev/polaris/.venv`

```

```{admonition} Upgrade <b>pip</b>
:class: warning
Be sure to upgrade `pip` itself as we've shown above -- otherwise, you may run into problems installing dependencies.
```

## Installing the latest Polaris release

From your [polaris dedicated python environment](#how-to-virtual-environment):
```{code-block} bash
(.polarisenv) $ pip install --upgrade pip # if not already done
(.polarisenv) $ pip install polaris-ml
```

## For developers: Installing the latest changes of Polaris

We push new releases to PyPi as appropriate, but work is continuing on Polaris all the time.  If you want to try out the very latest features (and help us with bug reports :grinning:), you can do that by following these steps.

From your [polaris dedicated python environment](#how-to-virtual-environment):
```{code-block} bash
(.polarisenv) $ pip install --upgrade pip # if not already done
(.polarisenv) $ git clone --recurse-submodules https://gitlab.com/librespacefoundation/polaris/polaris.git
(.polarisenv) $ cd polaris
(.polarisenv) $ pip install -e .
```

# Running your first analysis:  LightSail-2

Now you're ready to get started on your first analysis.  We've picked [LightSail-2](https://en.wikipedia.org/wiki/LightSail), a mission by The Planetary Society to [demonstrate the feasibility of solar sailing](https://www.planetary.org/articles/lightsail-2-extended-mission), to get started with, as there is a lot of data available.

![Sunrise from LightSail-2](https://planetary.s3.amazonaws.com/web/assets/pictures/20191121_C2-75-2019-09-28_100153_28_82_00942_292-0_pscam_dedistorted.jpg)

**LightSail 2 Orbital Sunrise:** The Sun rises over the horizon in this image from LightSail 2 captured on 28 September 2019. The sail appears curved due to the spacecraft's 185-degree fisheye camera lens. The image has been color corrected and some of the distortion has been removed. Licensed under [Creative Commons, BY-CC-NA](https://creativecommons.org/licenses/by-nc/3.0/) by The Planetary Society. ([Original source](https://www.planetary.org/space-images/lightsail-sun-on-horizon).)

## Fetching data with `polaris fetch`

We'll start by downloading a week's worth of data from [SatNOGS](https://satnogs.org).  Run this command:

```
polaris fetch \
    --start_date 2020-01-21 \
    --end_date 2020-01-28 \
    --cache_dir /tmp/LightSail-2 \
    LightSail-2 \
    /tmp/LightSail2-normalized_frames.json
```

Some background:

- The `--start_date` and `--end_date` arguments have been picked to favour a shorter download time, at the expense of having a decent amount of data.  You can experiment with different dates. (Note: if you create an account at db.satnogs.org, you can see [what telemetry they have available](https://db.satnogs.org/satellite/44420), and request data for download -- we'll show you how to import that later on.)

- The `--cache_dir` argument specifies where to cache downloaded data.

- The `LightSail-2` argument specifies a normalizer for the downloaded telemetry.  (Note: we'll show you later on how to analyze data without specifying a normalizer.)

- The final argument is the path to the output of `polaris fetch`: a JSON file with all the normalized data.

The final output will be a JSON file with a list of _frames_: individual telemetry transmissions from LightSail-2, captured by the SatNOGS network.

**Note:** If you have installed the latest version of Polaris as detailed above, you will *also* have space weather data for the time period we've fetched. ~~As of September 2020, this has not yet been released to PyPi.~~ Update: The latest PyPi release has space weather support.

**Note 2:** You can also batch download the frames from satnogs-db (as a csv) and run polaris fetch on that like so:
```
polaris fetch \
     --import_file ~/Downloads/44420-1363-20200824T065638Z-all.csv \
     --cache_dir /tmp/lightsail2 \
     LightSail-2 \
    /tmp/lightsail2/normalized_frames.json
```

**Note 3:** You can also get all the satellites supported by polaris by running command:
```
polaris fetch --list_supported_satellites
``` 
or 
```
polaris fetch -l
```

Let's take a look at how many frames we have:

```
$ jq '.frames | length' /tmp/LightSail2-normalized_frames.json
557
```

557 frames is not a lot; we'd want a great deal more data to do a decent analysis.  However, the small sample size makes it quicker to download, and _much_ quicker to analyze.

The next step is to analyze this data.

## Analyzing the data with `polaris learn`

Next, we run `polaris learn` to analyze the data.  This will use the [XGBoost library](https://xgboost.ai/) to create a dependency graph for the telemetry we've downloaded.  Run this command:

```
polaris learn \
    --force_cpu \
    --output_graph_file /tmp/LightSail2-graph.json \
    /tmp/LightSail2-normalized_frames.json
```

Some background:

- The `--force_cpu` option skips over any attempt to detect or use any GPU you might have.  If you have a supported GPU, and your drivers are working, you can try leaving out this argument -- it will speed things up immensely.

- The `--output_graph_file` argument specified where we'd like our dependency graph to be saved.  This will be the input for the next step.

- The final argument is the path to the normalized frame file we produced in the last step with `polaris fetch`.

Run times will depend on your hardware and the amount of data fetched.  On a 12-core, 2.6GHz i7 laptop, with 16 GB of RAM, `polaris learn` takes about two minutes.

## Visualizing the dependency graph with `polaris viz`

Finally, we're ready to look at our dependency graph.  Run this command:

```
polaris viz /tmp/LightSail2-graph.json
```

This will download some JavaScript dependencies, then open up a web server on your machine.  Open your browser and navigate to [http://localhost:8080](http://localhost:8080).  Here's what you should be seeing:

![Screenshot_2020-09-13_Polaris_-_tmp_LightSail2-graph_json](https://gitlab.com/librespacefoundation/polaris/polaris/-/wikis/uploads/f7be647ac8dbffcd6e758647bfe4102f/Screenshot_2020-09-13_Polaris_-_tmp_LightSail2-graph_json.png)

Let's examine the screen:

- The name of the satellite, `LightSail-2`, is in the top right-hand corner.

- The search bar is in the top left-hand corner.  We'll come back to this in a moment.

- The circles in the middle are the _nodes_ of the dependency graph.  Each one represents an element of telemetry.  The name of that element is displayed beside it.

- The lines connecting the nodes represent connections between those nodes -- that is, the model constructed by polaris learn has these two nodes varying in concert with each other.

You can navigate this graph:

- The scroll wheel on your mouse will zoom in or out.

- If you click-and-drag on empty space with your left mouse button, you can rotate the graph.

- If you click-and-drag on a node with your left mouse button, you can drag that node around.

- If you click-and-drag with your middle mouse button, you can drag the whole graph around.

- If you left-click on a particular node, you will zoom into that node.

When you're zoomed into a node, you'll see the lines connecting it to other nodes (unless it's a node that doesn't have a connection to anything else!).  You'll also see dots moving along those lines.  Those dots represent the _strength_ of the connection:  a faster-moving dot means a stronger connection.

Zoom back out again.  Click on the search bar, and type `py_tmp`.  You should see a list of node names come up as you type; you should end up with just `py_tmp:py_tmp/0` as a candidate.  Click on that, and you should zoom into that node.

Zoom back out again.  Click on the search bar, and type `tmp`.  Pause for a moment, and scroll through the list of nodes: these are all the nodes that have `tmp` somewhere in their name.  Now hit `Ctrl-enter` (that is, the control and enter key at the same time).  You should now see all of those nodes -- the ones with `tmp` in their name -- higlighted in the same colour.

Click on the search bar again, and go through that same process with each of the following:

- `mag`
- `cam`
- `pwr`

Make sure you hit `Ctrl-enter` after each one, but this time hit it a few times.  Note how each group is being highlighted in different colours, changing each time you hit `Ctrl-enter`.  This allows you to pick colours that work for you.

Looking at the display, we see a number of different things:

- There are different collections of nodes, each with their own colours.
- The groups of nodes have similar names; we can guess that they probably have similar functions.
- The groups are clustered in different ways, showing connections between these groups; this suggests that our model has found connections between different groups of functions.

The satellite operator, at this point, will be able to examine the graph with an eye toward finding connections between elements that may not have been noticed previously.

## (Optional) Converting dependency graph to another file format using `polaris convert`

Let's say we want to view the dependency graph in another program. This means the dependency graph must be converted into another file format supported by the program we're using. Currently, the dependency graph can be converted into `.gexf` file format which is supported by [Gephi](https://gephi.org/), the open graph tool.

To convert the current dependency graph (`.json`) to `.gexf`, we run this command:

```
polaris convert /tmp/LightSail2-graph.json /tmp/LightSail2-graph.gexf
```

Polaris will automatically detect which file format to convert to. In this case, because we specify `/tmp/LightSail2-graph.gexf`, Polaris will convert to GEXF file format.

Now we open the GEXF file in Gephi. After applying certain layouts and styles (Force Atlas, node & edge ranking), it will look something like this:

![gephi](https://gitlab.com/librespacefoundation/polaris/polaris/uploads/dc7032aa0d883a8239c2036a4c1e6382/preview1.png)


## Detect Anomalies in Telemetry Data using `polaris behave`

Another cool and useful thing that we can do with data that we fetched with `polaris fetch` is to map behaviour and (possible) find anomalies. Traditionally, most satellite operators set out-of-limit (OOL) thresholds to detect anomalies in spacecraft operations. This has the issue that small changes in a large number of parameters, which could affect the spacecraft behaviour, are not detected. `polaris behave` uses an auto-encoder model to accurately map different behaviour of the spacecraft and provide data regarding "breakpoints", places where there is a transition from one expected behaviour to another expected/unexpected behaviour. To see analysis, run this command:

```
polaris behave /tmp/LightSail2-normalized_frames.json \
    --output_file /tmp/LightSail2-anomaly_analysis.json
```

Some background:

- The first argument is the path to the input file of normalized frames generated by `polaris fetch` command

- The `--cache_dir` option specifies the directory to store additional files such as models used, normalizer, training data and test data

- The `--save_test_train_data` is here to give you option to choose whether to store test and train data or not because as sometimes test and train data can be quite large so the you might not want to store the input data

- The `--metrics_dir` option specifies the directory to save anomaly metrics

- The `--detector_config_file` option specifies the path to custom config file to specify the custom config file for parameters of detector 

- The `--csv_sep` option specifies the separator when input data is in CSV format.  Its default value is a comma (",").

- The `--output_file` provides the path in which output of the anomaly detector will be saved.

# Conclusion

Here we've taken you through the basics of the Polaris workflow:

- we've picked a satellite to work with;
- we've downloaded and normalized data captured by the SatNOGS network from that satellite;
- we've created a model of the connections between the telemetry elements;
- we've used that model to create a dependency graph;
- and we've displayed and examined the visualization of that graph.

We've also shown that we can convert the dependency graph to other file formats using `polaris convert`.

# Next Steps

- How to work with space weather data
- Using Polaris without a normalizer
- Downloading all the data SatNOGS has captured for a particular satellite
