import React, { memo, useEffect, useState } from "react";
import {
  Charts,
  ChartContainer,
  ChartRow,
  LineChart,
  AreaChart,
  LabelAxis,
} from "react-timeseries-charts";
import { TimeRange, TimeSeries } from "pondjs";
import { useTheme, Typography, makeStyles } from "@material-ui/core";
import Select from "react-select";
import makeAnimated from "react-select/animated";
import Resizable from "react-timeseries-charts/lib/components/Resizable";
import YAxis from "react-timeseries-charts/lib/components/YAxis";
import CustomSwitch from "../common/CustomSwitch";
import Baseline from "react-timeseries-charts/lib/components/Baseline";
import { color_pallete } from "../../config/colors";
import Legend from "react-timeseries-charts/lib/components/Legend";
import { toShortFormat } from "../../utils/helper";
const animatedComponents = makeAnimated();

const useStyles = makeStyles((theme) => ({
  container: { width: "100%", marginTop: "5rem" },
  switchContainer: {
    marginBottom: "2rem",
    display: "flex",
    justifyContent: "flex-end",
    width: "100%",
  },
  markerContainer: {
    position: "relative",
  },
}));

const labelStyle = (theme) => ({
  axis: {
    fontSize: 11,
    textAnchor: "left",
    fill: theme.palette.text.primary,
  },
  label: {
    fontSize: 16,
    textAnchor: "middle",
    fill: theme.palette.text.primary,
  },
  values: { fill: theme.palette.text.primary, stroke: "rgb(221, 221, 221)" },
});

const customSelectStyles = (theme) => ({
  container: (provided, state) => ({
    ...provided,
    marginTop: "3rem",
    marginBottom: "2rem",
  }),
  menu: (provided, state) => ({
    ...provided,
    color: theme.palette.text.primary,
    backgroundColor: theme.palette.background.default,
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: theme.palette.background.default,
    "&:hover": {
      backgroundColor: "#cccccc33",
    },
    "&:active": {
      backgroundColor: "#cccccc33",
    },
  }),

  control: (provided, state) => ({
    ...provided,
    backgroundColor: theme.palette.background.default,
    color: theme.palette.text.primary,
    border: "2px solid #00FFF0",
    "&:hover": {
      border: "2px solid #00FFF0",
    },
  }),
  clearIndicator: (provided, state) => ({
    ...provided,
    color: "#00FFF0",
    "&:hover": {
      color: "#00FFF0",
    },
  }),
  dropdownIndicator: (provided, state) => ({
    ...provided,
    color: "#00FFF0",
    "&:hover": {
      color: "#00FFF0",
    },
  }),

  multiValue: (provided, state) => {
    return {
      ...provided,
      background: `#00FFF07F`,
      marginRight: "1rem",
      paddingLeft: "3px",
    };
  },
  multiValueLabel: () => ({}),
});

const customLegendStyles = (columnsToCompare) => {
  let legendStyle = {};
  columnsToCompare.forEach((col, idx) => {
    legendStyle[col] = {
      symbol: {
        normal: {
          stroke: "#00fff0",
          fill: color_pallete[idx % color_pallete.length],
        },
      },
    };
  });
  return legendStyle;
};

const getData = (jsonData, columnsToCompare) => {
  const points = {};
  columnsToCompare.forEach((col) => (points[col] = []));
  jsonData.data.timestamps.forEach((stamp, idx) => {
    columnsToCompare.forEach((col) => {
      let temp = new Date(stamp);
      let time = temp.getTime();
      points[col].push([
        time,
        jsonData.data.values[col].individual_values[idx],
      ]);
    });
  });

  columnsToCompare.forEach((col) => {});
  const data = {};

  let maxVal = Number.MIN_VALUE;
  let minVal = Number.MAX_VALUE;

  for (let col of columnsToCompare) {
    const series = new TimeSeries({
      name: col,
      columns: ["time", col],
      points: points[col],
    });
    if (!Boolean(data[col])) data[col] = {};
    data[col].series = series;
    data[col].avg = parseInt(series.avg(col).toFixed(2), 10);
    maxVal = Math.max(maxVal, parseInt(series.max(col).toFixed(2), 10));
    minVal = Math.min(minVal, parseInt(series.min(col).toFixed(2), 10));
  }
  for (let col of columnsToCompare) {
    data[col].max = maxVal;
    data[col].min = minVal;
  }
  return data;
};

const handleTrackerChanged = (setTracker, setTrackerX) => (time, scale) => {
  setTracker(time);
  if (time) {
    const date = new Date(time);
    const trackerX = scale(date);
    setTrackerX(trackerX);
  }
};

const CompareGraph = memo(
  ({ jsonData = {} }) => {
    const [data, setData] = useState({});
    const [timeRange, setTimeRange] = useState(
      new TimeRange([75 * 60 * 1000, 125 * 60 * 1000])
    );
    const [columnsToCompare, setColumnsToCompare] = useState([]);
    const [mode, setMode] = useState("stacked");
    const [tracker, setTracker] = useState(null);
    const [trackerX, setTrackerX] = useState(0);
    const theme = useTheme();
    const [isLoading, setIsLoading] = useState(false);
    const [minTime, setMinTime] = useState();
    const [maxTime, setMaxTime] = useState();
    const [randomColors, setRandomColors] = useState({});
    const legendStyle = customLegendStyles(columnsToCompare);
    const classes = useStyles();
    const trackerInfoValues = columnsToCompare
      .map((col) => {
        if (data[col] === undefined) return;
        let series = data[col].series.crop(timeRange);

        let v = "--";
        if (tracker) {
          const i = series.bisect(new Date(tracker));
          const vv = series.at(i).get(col);
          if (vv) {
            v = vv.toFixed(2);
          }
        }

        const label = col;
        const value = `${v}`;

        return { label, value };
      })
      .filter(Boolean);

    useEffect(() => {
      if (jsonData.data && columnsToCompare.length > 0) {
        const newData = getData(jsonData, columnsToCompare);
        setData(newData);
        setTimeRange(newData[columnsToCompare[0]].series.timerange());
        setMinTime(newData[columnsToCompare[0]].series.timerange().begin());
        setMaxTime(newData[columnsToCompare[0]].series.timerange().end());

        const newColors = {};
        columnsToCompare.forEach((col, idx) => {
          newColors[col] = color_pallete[idx % color_pallete.length];
        });
        setRandomColors(newColors);
      }
    }, [jsonData, columnsToCompare]);

    useEffect(() => {
      setIsLoading(false);
    }, [data]);

    const renderChannelChart = () => {
      const rows = [];
      if (!Boolean(data)) return rows;
      columnsToCompare.forEach((col, idx) => {
        const series = data[col].series;
        rows.push(
          <ChartRow height="150" key={`row-${col}`}>
            <LabelAxis
              id={`${col}_axis`}
              label={col}
              min={data[col].min}
              max={data[col].max}
              style={labelStyle(theme)}
              width={200}
              type="linear"
              format=",.1f"
            />
            <Charts>
              <AreaChart
                key={`line-${col}`}
                axis={`${col}_axis`}
                series={series}
                columns={{ up: [col] }}
                style={{
                  [col]: {
                    line: {
                      normal: {
                        stroke: "#00FFF0",
                        fill: "none",
                        strokeWidth: 1,
                      },
                    },
                    area: {
                      normal: {
                        fill: randomColors[col],
                        stroke: "none",
                        opacity: 0.75,
                      },
                    },
                  },
                }}
                interpolation="curveBasis"
                fillOpacity={0.4}
              />
            </Charts>
          </ChartRow>
        );
      });
      return (
        <ChartContainer
          timeRange={timeRange}
          width={1000}
          showGrid={false}
          enablePanZoom
          trackerPosition={tracker}
          onTrackerChanged={handleTrackerChanged(setTracker, setTrackerX)}
          onTimeRangeChanged={(timerange) => setTimeRange(timerange)}
          maxTime={maxTime}
          minTime={minTime}
        >
          {rows}
        </ChartContainer>
      );
    };

    const renderMultiAxisChart = () => {
      const charts = [];
      columnsToCompare.forEach((col, idx) => {
        let series = data[col].series;

        charts.push(
          <LineChart
            key={`line-${col}`}
            axis={`${columnsToCompare[0]}_axis`}
            series={series}
            columns={[col]}
            breakLine
            style={{
              [col]: {
                normal: {
                  stroke: randomColors[col],
                  strokeWidth: 3,
                },
              },
            }}
          />
        );
      });

      return (
        <ChartContainer
          timeRange={timeRange}
          trackerPosition={tracker}
          onTrackerChanged={handleTrackerChanged(setTracker, setTrackerX)}
          trackerShowTime
          enablePanZoom
          maxTime={maxTime}
          minTime={minTime}
          onTimeRangeChanged={(timerange) => setTimeRange(timerange)}
        >
          <ChartRow height="200">
            <YAxis
              id={`${columnsToCompare[0]}_axis`}
              key={`${columnsToCompare[0]}_axis`}
              min={data[columnsToCompare[0]].min}
              max={data[columnsToCompare[0]].max}
              width={70}
              type="linear"
            />
            <Charts>
              {charts}
              <Baseline value={0} axis={`${columnsToCompare[0]}_axis`} />
            </Charts>
          </ChartRow>
        </ChartContainer>
      );
    };

    return (
      <div className={classes.container}>
        {jsonData.data && (
          <Typography variant="h4">Individual Telemetry Graphs</Typography>
        )}
        {jsonData.data && (
          <Select
            styles={customSelectStyles(theme)}
            options={Object.keys(jsonData.data.values).map((col) => ({
              label: col,
              value: col,
            }))}
            closeMenuOnSelect={false}
            components={animatedComponents}
            onChange={(col) => {
              setIsLoading(true);
              setColumnsToCompare(col.map((el) => el.value));
            }}
            isMulti
          />
        )}
        {jsonData.data && columnsToCompare.length > 0 && (
          <div>
            <Typography component="div" className={classes.switchContainer}>
              <CustomSwitch
                leftLabel="stacked"
                rightLabel="merged"
                isChecked={mode === "multi-axis"}
                onChange={(check) => {
                  setMode(check ? "multi-axis" : "stacked");
                }}
              />
            </Typography>

            <div className={classes.markerContainer}>
              <div style={{ position: "absolute", left: trackerX }}>
                {tracker && (
                  <>
                    <div>{toShortFormat(tracker)}</div>
                    {trackerInfoValues.map((el, idx) => (
                      <div
                        style={
                          mode === "stacked"
                            ? {
                                top: `${idx * 150 + 16}px`,
                                position: "absolute",
                              }
                            : {}
                        }
                      >
                        {el.label} {el.value}
                      </div>
                    ))}
                  </>
                )}
              </div>
            </div>
            <Resizable>
              {isLoading ? (
                <div>Loading</div>
              ) : mode === "multi-axis" ? (
                renderMultiAxisChart()
              ) : (
                renderChannelChart()
              )}
            </Resizable>
            {Object.keys(legendStyle).length > 0 &&
              Object.keys(legendStyle).length === columnsToCompare.length && (
                <Legend
                  categories={columnsToCompare.map((col) => ({
                    key: `${col}`,
                    label: col,
                  }))}
                  style={legendStyle}
                  type="dot"
                />
              )}
          </div>
        )}
      </div>
    );
  },
  (prev, next) => {
    return (
      prev.jsonData === next.jsonData &&
      prev.columnsToCompare === next.columnsToCompare
    );
  }
);

export default CompareGraph;
