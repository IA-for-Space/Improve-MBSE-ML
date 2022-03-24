import { makeStyles } from "@material-ui/core";
import React, { useState, memo } from "react";
import { useEffect } from "react";
import { toShortFormat } from "../../utils/helper";
import { getIndividualEvents } from "./ContributionGraph";
import quote from "../../assets/images/quote.png";

const useStyles = makeStyles((theme) => ({
  container: { position: "relative" },
  header: { fontSize: "2rem" },
  unorderedList: {},
  listItems: {
    lineHeight: "3rem",
    listStyleType: "none",

    "&::before": {
      content: "'\\2022'",
      fontSize: "2rem",
      verticalAlign: "middle",
      color: theme.palette.text.primary,
      opacity: "0.6",
      marginRight: "2rem",
      transition: "0.5s ease-in"
    },
  },
  quote: {
    position: "absolute",
    top: "4.5rem",
  },
}));

const getTotalFrames = (jsonData) => jsonData.metadata.total_frames;

const getTotalConstantTelemetry = (jsonData) =>
  Object.values(jsonData.metadata.analysis.column_tags).filter(
    (el) => el === "constant"
  ).length;

const getTotalTelemetry = (jsonData) =>
  Object.values(jsonData.metadata.analysis.column_tags).length;

const Highlights = memo(({ jsonData }) => {
  const classes = useStyles();
  const [eventHighlights, setEventHighlights] = useState({
    totalEvents: 0,
    totalTelemetryResponsible: 0,
    totalFrames: 0,
    totalConstantTelemetry: 0,
    totalTelemetry: 0,
    timeRange: { startDate: new Date(), endDate: new Date() },
  });

  useEffect(() => {
    if (jsonData.data) {
      const totalEvents = jsonData.data.events.length;
      const individualEventsDetected = getIndividualEvents(jsonData);
      const startDate = new Date(jsonData.data.timestamps[0]);
      const endDate = new Date(
        jsonData.data.timestamps[jsonData.data.timestamps.length - 1]
      );
      setEventHighlights((highlights) => ({
        ...highlights,
        totalEvents,
        totalTelemetryResponsible: Object.keys(individualEventsDetected).length,
        timeRange: { startDate, endDate },
      }));
    }
    if (jsonData.metadata) {
      const totalFrames = getTotalFrames(jsonData);
      const totalConstantTelemetry = getTotalConstantTelemetry(jsonData);
      const totalTelemetry = getTotalTelemetry(jsonData);
      setEventHighlights((highlights) => ({
        ...highlights,
        totalFrames,
        totalConstantTelemetry,
        totalTelemetry,
      }));
    }
  }, [jsonData]);

  return (
    <div className={classes.container}>
      <img
        src={quote}
        alt="quote symbol"
        width="32"
        className={classes.quote}
      />
      <header className={classes.header}>Highlights</header>
      <ul className={classes.unorderedList}>
        <li className={classes.listItems}>
          Total {eventHighlights.totalEvents} Anomaly Detected
        </li>
        <li className={classes.listItems}>
          Total {eventHighlights.totalTelemetryResponsible} Telemetries were
          responsible for the anomalies{" "}
        </li>
        <li className={classes.listItems}>
          Time Range from {toShortFormat(eventHighlights.timeRange.startDate)}{" "}
          to {toShortFormat(eventHighlights.timeRange.endDate)}
        </li>
        <li className={classes.listItems}>
          Total {eventHighlights.totalConstantTelemetry} out of{" "}
          {eventHighlights.totalTelemetry} telemetry are constant
        </li>
        <li className={classes.listItems}>
          Total {eventHighlights.totalFrames} Frames
        </li>
      </ul>
    </div>
  );
});

export default Highlights;
