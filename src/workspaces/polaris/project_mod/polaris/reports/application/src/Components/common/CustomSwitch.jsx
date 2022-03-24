import React from "react";
import { makeStyles, Switch, Grid } from "@material-ui/core";

const useStyles = makeStyles((theme) => ({
  container: { width: "max-content" },
  root: {
    width: 48,
    height: 23,
    padding: "0 0 7px 0",
    boxSizing: "content-box",
    margin: theme.spacing(1),
  },
  switchBase: {
    padding: 1,
    "&$checked": {
      transform: "translateX(16px)",
      "& + $track": {
        backgroundColor: theme.palette.background.default,
        opacity: 1,
      },
    },
    "&$focusVisible $thumb": {
      border: "1px solid #00FFF0",
    },
  },
  thumb: {
    width: 28,
    height: 28,
    color: "transparent",
    border: "2px solid #00FFF0",
    boxSizing: "border-box",
  },
  track: {
    borderRadius: 28 / 2,
    border: "2px solid #00FFF0",
    backgroundColor: "transparent",
    opacity: 1,
    margin: "2px",
    transition: theme.transitions.create(["background-color", "border"]),
  },
  checked: {},
  focusVisible: {},
}));

const CustomSwitch = ({
  leftLabel = "",
  rightLabel = "",
  isChecked = false,
  onChange = () => {},
}) => {
  const classes = useStyles();
  return (
    <Grid
      component="label"
      container
      alignItems="center"
      justifyContent="flex-end"
      spacing={1}
      className={classes.container}
    >
      <Grid item>{leftLabel}</Grid>
      <Grid item>
        <Switch
          focusVisibleClassName={classes.focusVisible}
          disableRipple
          classes={{
            root: classes.root,
            switchBase: classes.switchBase,
            thumb: classes.thumb,
            track: classes.track,
            checked: classes.checked,
          }}
          checked={isChecked}
          onChange={(e) => {
            onChange(e.target.checked);
          }}
          name="checkedB"
        />
      </Grid>
      <Grid item>{rightLabel}</Grid>
    </Grid>
  );
};

export default CustomSwitch;
