import React, { useEffect, useState, memo } from "react";
import Chart from "react-apexcharts";
import { getCommon } from "../../utils/helper";
import { useTheme } from "@material-ui/core";

export const getIndividualEvents = (jsonData) => {
  const columns = Object.keys(jsonData.data.values);
  const events = jsonData.data.events.map(
    (evt) => jsonData.data.timestamps[evt]
  );
  const eventData = {};
  columns.forEach((col) => {
    let common = getCommon(
      events,
      jsonData.data.values[col]["individual_events_detected"]
    );
    if (common.length > 0) {
      eventData[col] = common.length;
    }
  });
  return eventData;
};

const getEventOptions = (eventData, theme) => ({
  chart: {
    type: "polarArea",
    foreColor: theme.palette.text.primary,
  },
  labels: Object.keys(eventData),
  stroke: {
    width: 1,
    colors: undefined,
  },
  fill: {
    opacity: 1,
  },
  plotOptions: {
    polarArea: {
      rings: {
        strokeWidth: 0,
      },
      spokes: {
        strokeWidth: 0,
      },
    },
  },
  yaxis: { show: false },
  legend: {
    position: "bottom",
  },
});

const ContributionGraph = memo(({ jsonData }) => {
  const [data, setData] = useState([]);
  const [options, setOptions] = useState({});
  const theme = useTheme();
  useEffect(() => {
    if (jsonData.data) {
      const eventData = getIndividualEvents(jsonData);
      const newEventOptions = getEventOptions(eventData, theme);
      setData(eventData);
      setOptions(newEventOptions);
    }
  }, [jsonData, jsonData.data, theme]);

  return (
    <div>
      <h2>Contributions of telemetry in events</h2>
      <Chart
        options={options}
        series={Object.values(data)}
        type="polarArea"
        width={600}
      />
    </div>
  );
});

export default ContributionGraph;
