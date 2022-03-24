import { useTheme } from "@material-ui/core";
import React, { useEffect, useState } from "react";
import { useRef } from "react";
import { useContext } from "react";
import Chart from "react-apexcharts";
import { StateContext } from "../../App";
import { toShortFormat } from "../../utils/helper";

const getTimeStamps = (timestrings) =>
  timestrings.map((time) => {
    let temp = new Date(time);
    return temp.getTime();
  });

const parseData = (timestamps, jsonData) => {
  const columns = Object.keys(jsonData.data.values);
  const newData = [
    {
      name: "positive",
      data: [],
    },
    { name: "negative", data: [] },
  ];
  for (let i = 0; i < timestamps.length; i++) {
    let pos = 0;
    let neg = 0;
    columns.forEach((col) => {
      let temp = jsonData.data.values[col]["individual_values"][i];
      if (temp > 0) {
        pos += temp;
      } else {
        neg += temp;
      }
    });
    newData[0].data.push({ x: timestamps[i], y: pos.toFixed(2) });
    newData[1].data.push({ x: timestamps[i], y: neg.toFixed(2) });
  }
  return newData;
};

const getOptions = (theme) => ({
  chart: {
    type: "area",
    redrawOnParentResize: true,
    foreColor: theme.palette.text.primary,
    background: theme.palette.background.default,
    zoom: {
      autoScaleYaxis: true,
    },
    toolbar: {
      theme: "dark",
      autoSelected: "pan",
    },
  },
  colors: ["#00BAEC"],
  stroke: {
    width: 3,
  },
  grid: {
    borderColor: theme.palette.custom.stroke,
    clipMarkers: false,
  },
  dataLabels: {
    enabled: false,
  },
  fill: {
    type: "gradient",
    gradient: {
      type: "horizontal",
      shadeIntensity: 0.5,
      enabled: true,
      stops: [0, 50, 100],
      opacityFrom: 0.7,
      opacityTo: 0.9,
    },
  },
  legend: {
    show: false,
  },
  xaxis: {
    type: "datetime",
  },
});

const getTooltipOptions =
  (events) =>
  ({ series, seriesIndex, dataPointIndex, w }) => {
    const date = new Date(w.globals.seriesX[seriesIndex][dataPointIndex]);
    return `
      <div
        class="apexcharts-tooltip-title"
        style="font-family: Helvetica, Arial, sans-serif; font-size: 12px;"
      >
      ${toShortFormat(date)}
      </div>
      <div
        class="apexcharts-tooltip-series-group apexcharts-active"
        style="order: 1; display: flex;"
      >
        <span
          class="apexcharts-tooltip-marker"
          style="background-color: rgb(0, 186, 236);"
        ></span>
        <div
          class="apexcharts-tooltip-text"
          style="font-family: Helvetica, Arial, sans-serif; font-size: 12px;"
        >
          <div class="apexcharts-tooltip-y-group">
            <span class="apexcharts-tooltip-text-y-label">Max: </span>
            <span class="apexcharts-tooltip-text-y-value">${
              series[0][dataPointIndex]
            }</span>
          </div>
        </div>
      </div>
      <div
        class="apexcharts-tooltip-series-group apexcharts-active"
        style="order: 2; display: flex;"
      >
        <span
          class="apexcharts-tooltip-marker"
          style="background-color: rgb(0, 186, 236);"
        ></span>
        <div
          class="apexcharts-tooltip-text"
          style="font-family: Helvetica, Arial, sans-serif; font-size: 12px;"
        >
          <div class="apexcharts-tooltip-y-group">
            <span class="apexcharts-tooltip-text-y-label">Min: </span>
            <span class="apexcharts-tooltip-text-y-value">${
              series[1][dataPointIndex]
            }</span>
          </div>
        </div>
      </div>
      <div
        class="apexcharts-tooltip-series-group apexcharts-active"
        style="order: 3; display: flex;"
      >
        <span
          class="apexcharts-tooltip-marker"
          style="background-color: rgb(0, 186, 236);"
        ></span>
        <div
          class="apexcharts-tooltip-text"
          style="font-family: Helvetica, Arial, sans-serif; font-size: 12px;"
        >
          <div class="apexcharts-tooltip-y-group">
            <span class="apexcharts-tooltip-text-y-label">Event Occured: </span>
            <span class="apexcharts-tooltip-text-y-value">${
              events.includes(dataPointIndex) ? "Yes" : "No"
            }</span>
          </div>
        </div>
      </div>
    `;
  };

const getXAxisAnnotations = (json_data, timestamps) =>
  json_data.data.events.map((event) => {
    return {
      x: timestamps[event],
      strokeDashArray: 0,
      borderColor: "#FF0080",
    };
  });

const StackedGraph = ({ jsonData }) => {
  const [data, setData] = useState([
    {
      name: "positive",
      data: [],
    },
    { name: "negative", data: [] },
  ]);
  const state = useContext(StateContext);
  const currentTheme = state.currentTheme;
  const theme = useTheme();
  const [options, setOptions] = useState(getOptions(theme));
  const containerRef = useRef(null);

  useEffect(() => {
    let timestamps = [];
    if (jsonData.data && jsonData.data.timestamps) {
      timestamps = getTimeStamps(jsonData.data.timestamps);
    }
    if (jsonData.data) {
      const newData = parseData(timestamps, jsonData);
      setData(newData);

      const newOptions = {
        ...getOptions(theme),
        annotations: {
          xaxis: getXAxisAnnotations(jsonData, timestamps),
        },
        tooltip: {
          theme: currentTheme,
          custom: getTooltipOptions(jsonData.data.events),
        },
        title: {
          text: `${jsonData.metadata.satellite_name} analysis with normalized values`,
          align: "left",
          margin: 10,
          offsetX: 10,
          offsetY: 0,
          floating: false,
          style: {
            fontSize: "24px",
            fontWeight: "bold",
          },
        },
      };
      setOptions(newOptions);
    }
  }, [jsonData, jsonData.data, currentTheme, theme]);

  return (
    <div
      style={{
        width: "100%",
        marginTop: "3rem",
        display: "flex",
        justifyContent: "center",
      }}
      ref={containerRef}
    >
      <Chart
        options={options}
        series={data}
        type="area"
        width={
          containerRef.current ? containerRef.current.clientWidth * 0.9 : "1000"
        }
      />
    </div>
  );
};

export default StackedGraph;
