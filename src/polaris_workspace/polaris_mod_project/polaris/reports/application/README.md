# Polaris Reports

This repo holds the React application that displays Polaris behaviour reports; it takes the JSON file produced by `polaris behave`, and generates an in-browser report for the user.

The frameworks used are [React](https://reactjs.org) and [CreateReactApp](https://create-react-app.dev/).

## Prerequisites for Installation

- [Node](https://nodejs.org/)
- [Yarn](https://yarnpkg.com/)

Consult your operating system/distro documentation for information on how to install these packages.

## Setup Instructions

1. Clone the repo:

```
git@gitlab.com:librespacefoundation/polaris/polaris-reports.git
```

2. Enter in the Repo folder:

```
cd polaris-reports
```

3. Install the dependencies:

```
yarn
```

4. Start the development server:

```
yarn start
```

5. Open [http://localhost:3000](http://localhost:3000) to view the application in the browser. This will display a report for LightSail-2, using the file `public/analysis.json`.

## How to?

###  Display a report for another satellite

1. Use `polaris behave` to generate a new analysis file.  See [our documentation](https://docs.polarisml.space/en/latest/using/getting_started_with_polaris.html#detect-anomalies-in-telemetry-data-using-polaris-behave) for instructions on how to do this.

2. Copy the generated file to `public/analysis.json` if you are using the development server, or to the build folder if you are using the production server.  **Note:** Currently the file name is hardcoded in `src/views/HomeView.jsx`.

## Working with yarn

### Local development

To start the aplication in development mode, run:

```
yarn start
```

Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will automatically reload if you make edits. You will also see any lint errors in the terminal.

### Building for deployment

To build the application for deployment, run:

```
yarn build
```

This will:

- build the app in the `build` folder;
- bundle React in production mode;
- optimize the build for the best performance;
- and minify and checksum files.

To deploy to a webserver, just copy the `build` directory:

```
rsync -avr ./build/. webserver.example.com:/var/www/html/polaris_reports/
```

See the [deployment chapter of the Create React App documentation](https://create-react-app.dev/docs/deployment) for more information.

## Learn More

To learn React, check out the [React documentation](https://reactjs.org/).

To learn about CreateReactApp, check out the [Create React App documentation](https://create-react-app.dev/).
