import { makeStyles, Typography } from "@material-ui/core";
import React, { memo, useEffect, useState } from "react";
import CompareGraph from "../Components/Home/CompareGraph";
import ContributionGraph from "../Components/Home/ContributionGraph";
import Highlights from "../Components/Home/Highlights";
import StackedGraph from "../Components/Home/StackedGraph";

const useStyles = makeStyles((theme) => ({
  container: {
    display: "flex",
    justifyContent: "spaceEvenly",
    alignItems: "center",
    backgroundColor: theme.palette.background.default,
    color: theme.palette.text.primary,
    transition: "0.5s ease-in",
    flexDirection: "column",
  },
}));

const HomeView = memo(() => {
  const [jsonData, setJsonData] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const classes = useStyles();
  useEffect(() => {
    setIsLoading(true);
    fetch("analysis.json")
      .then((res) => res.json())
      .then((data) => {
        setIsLoading(true);
        setJsonData(data);
      });
  }, []);
  return (
    <div className={classes.container}>
      <Typography variant="h3" align="center">
        {jsonData.metadata &&
          `${jsonData.metadata.satellite_name} Behave Analysis`}
      </Typography>
      <div
        style={{
          display: "flex",
          justifyContent: "space-around",
          alignItems: "center",
          marginTop: "5rem",
          marginBottom: "3rem",
          width: "100%",
        }}
      >
        <ContributionGraph jsonData={jsonData} />
        <Highlights jsonData={jsonData} />
      </div>
      <StackedGraph jsonData={jsonData} />
      {jsonData.data && (
        <CompareGraph
          jsonData={jsonData}
          columnsToCompare={["daughter_atmp", "atmelpwrcurr"]}
        />
      )}
    </div>
  );
});

export default HomeView;
