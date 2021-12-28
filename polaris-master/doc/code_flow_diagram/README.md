# Polaris Code-Flow Diagram Designed with PlantUML

## Online Rescources
- [PlantUML Website](http://plantuml.com/en/)
- A good source for learning more about how to use plantUML: [Reference Guide ](https://deepu.js.org/svg-seq-diagram/Reference_Guide.pdf)

## Generating the Diagram
There are multiple ways to generate these kind of diagrams:
- online generator
    - [PlantUML Server](http://www.plantuml.com/plantuml/uml/SyfFKj2rKt3CoKnELR1Io4ZDoSa70000)
    - [PlantText](https://www.planttext.com)
- offline generator based on Java [downloadable here](https://plantuml.com/download)
    - plugins for editors like Atom, VSCode
    - install or download plantuml and run one of the following command to generate a PNG file inside /tmp/
    ```
        # plantuml -tpng -o /tmp/ polaris_code_flow.plantuml
        # java -jar ~/downloads/plantuml.1.2020.0.jar -tpng -o /tmp/ polaris_code_flow.plantuml
    ```

### Online Generator

**Disclaimer:** 
plantUML seems to have a problem with diagrams above a certain width, so to avoid this it is crucial to origanize the diagram by including direction to the arrows.

**Step1:** Open one of the online generators
**Step2:** Copy the code from the ‘.plantuml‘ file into the generator
**Step3:** Hit *Submit* or *Refresh* to generate your diagram.

## Quick Tutorial
#### Titel in a **bold font** and colour
```
title
  <b><color:gold>The Polaris Code Flow</b>
end title
```
### Creating a class *analysis* inside the category *learn* including a description and a method *cross_correlate()*
```
class learn.analysis{
Module to launch different data analysis.
  # cross_correlate()
}
```

#### Linking Two Parts with a blue arrow and blue description
```
fetch.data_fetch_decoder --right--> learn.analysis #line:Blue;header:blue : <color:blue>normalized_frames.json
```
