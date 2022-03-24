import { makeStyles } from "@material-ui/core";
import React, { useContext, memo } from "react";
import { DispatchContext, StateContext } from "../../App";
import logo from "../../assets/images/polaris_logo.png";
import ThemeSwitch from "./ThemeSwitch";

const useStyles = makeStyles((theme) => ({
  container: {
    display: "flex",
    height: "80px",
    justifyContent: "space-between",
    width: "100%",
    alignItems: "center",
    position: "fixed",
    zIndex: 90,
    top: 0,
    left: 0,
    backgroundColor: theme.palette.background.default,
    color: theme.palette.text.primary,
    transition: "0.5s ease-in",
    animation: "slide-in 0.6s ease",
  },
  stacked: {
    boxShadow: "0 1px 9px grey",
    height: "60px",
    transition: "0.5s ease-in",
  },

  hidden: {
    transform: "translateY(-100%)",
  },
  logo: {
    marginLeft: "1rem",
  },
  header: {
    fontSize: "3rem",
  },
  themeSwitch: {
    marginRight: "1rem",
  },
}));

const Navbar = memo(({ stacked, hidden }) => {
  const dispatch = useContext(DispatchContext);
  const state = useContext(StateContext);
  const { currentTheme } = state;
  const classes = useStyles();
  const cssClasses = [classes.container];

  if (stacked) {
    cssClasses.push(classes.stacked);
  }

  if (hidden) {
    cssClasses.push(classes.hidden);
  }
  return (
    <>
      <div className={cssClasses.join(" ")}>
        <a target="_blank" href="https://polarisml.space/" rel="noreferrer">
          <img src={logo} height={35} className={classes.logo} alt="logo" />
        </a>
        <div className={classes.header}>Polaris Reports</div>
        <div className={classes.themeSwitch}>
          <ThemeSwitch
            checked={currentTheme === "light"}
            themeChange={() => {
              dispatch({
                type: "changeTheme",
                payload: currentTheme === "light" ? "dark" : "light",
              });
            }}
          />
        </div>
      </div>
    </>
  );
});

export default Navbar;
