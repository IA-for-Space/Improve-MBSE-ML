import { makeStyles } from "@material-ui/core";
import React from "react";

const useStyles = makeStyles(() => ({
  container: {
    cursor: "pointer",
  },
  checkbox: {
    display: "none",
  },
  extra: (checked) => ({
    borderRadius: "50%",
    width: "36px",
    height: "36px",
    position: "relative",
    boxShadow: checked
      ? "inset 32px -32px 0 0 #fff"
      : "inset 14px -14px 0 0 var(--color-toggle-dark, #000)",
    transform: checked ? "scale(.5) rotate(0deg)" : "scale(1) rotate(-2deg)",
    transition: checked
      ? "transform .3s ease .1s, box-shadow .2s ease 0s"
      : "box-shadow .5s ease 0s, transform .4s ease .1s",

    "&::before": {
      content: '""',
      width: "inherit",
      height: "inherit",
      borderRadius: "inherit",
      position: "absolute",
      left: 0,
      top: 0,
      background: checked ? "var(--color-toggle-light, #eee)" : "inherit",
      transition: checked ? "background .3s ease .1s" : "background .3s ease",
    },
    "&::after": {
      content: '""',
      width: "8px",
      height: "8px",
      borderRadius: "50%",
      margin: "-4px 0 0 -4px",
      position: "absolute",
      top: "50%",
      left: "50%",
      boxShadow:
        "0 -23px 0 var(--color-toggle-light, #eee), 0 23px 0 var(--color-toggle-light, #eee), 23px 0 0 var(--color-toggle-light, #eee), -23px 0 0 var(--color-toggle-light, #eee), 15px 15px 0 var(--color-toggle-light, #eee), -15px 15px 0 var(--color-toggle-light, #eee), 15px -15px 0 var(--color-toggle-light, #eee), -15px -15px 0 var(--color-toggle-light, #eee)",
      transform: checked ? "scale(1.5)" : "scale(0)",
      transition: checked ? "transform .5s ease .15s" : "all .3s ease",
    },
  }),
}));

const ThemeSwitch = ({ checked: isDay, themeChange }) => {
  const classes = useStyles(isDay);
  return (
    <label
      className={classes.container}
      title={!isDay ? "Activate light mode" : "Activate dark mode"}
    >
      <input
        className={classes.checkbox}
        type="checkbox"
        id="toggle-checkbox"
        name="toggle-checkbox"
        onChange={themeChange}
        checked={isDay}
      />
      <div className={classes.extra}></div>
    </label>
  );
};

export default ThemeSwitch;
