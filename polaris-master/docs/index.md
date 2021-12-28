# Welcome to Polaris's documentation!

**Polaris** is an open source (LGPLv3) project that analyzes satellite telemetry using machine learning.  It is a project of the Libre Space Foundation.

**Want to get started quickly?** Have a look at [Getting started with Polaris](using/getting_started_with_polaris).


Polaris has a number of different commands:

- `polaris fetch` will download and normalize satellite telemetry from the SatNOGS network (or you can import your own).
- `polaris learn` will analyze the telemetry, produce a model of the connections between telemetry components, and save a dependency graph for visualization.
- `polaris viz` is an interactive, browser-based 3D visualization of that dependency graph.
- `polaris convert` will convert graph output from `polaris learn` to another file format (like `.gexf`).
- `polaris behave` will detect anomalies in telemetry data and produce a json report of all the data and anomaly produced.

* For more details about how Polaris works, see [Overview](Overview), or watch our presentation at the [2020 Cubesat Developer's Workshop](https://www.youtube.com/watch?v=Jp7GuA_zjlA).

* Would you like to join the community?  We'd love to get to know you! [Join our chat room here](https://app.element.io/#/room/#polaris:matrix.org) and introduce yourself.

* If you'd like to work with us through the Google Summer of Code, have a look at [our notes for GSoC applicants](https://gitlab.com/librespacefoundation/polaris/polaris/-/wikis/Notes-for-Summer-of-Code-applicants).

* For other presentations on Polaris, see [Presentations on Polaris: Past and Future](https://gitlab.com/librespacefoundation/polaris/polaris/-/wikis/Presentations-on-Polaris:-Past-and-Future).

## Site contents

```{toctree}
---
maxdepth: 1
caption: Quick start
---
What's a satellite? <using/getting_started_with_polaris>
```

```{toctree}
---
maxdepth: 1
caption: Overview
---
overview.md
```

```{toctree}
---
maxdepth: 1
caption: Contributing
---
Gitlab repo <https://gitlab.com/librespacefoundation/polaris/polaris>
Chat <https://app.element.io/#/room/#polaris:matrix.org>
```
