import { makeStyles } from "@material-ui/core";
import React from "react";

const useStyles = makeStyles((theme) => ({
  ScrollToTop: {
    position: "fixed",
    bottom: "120px",
    right: "50px",
    color: " #7fffd4",
    border: "1px solid #7fffd4",
    padding: "15px 20px",
    borderRadius: " 4px",
    opacity: "0.8",
    zIndex: 500,
    animation: "$slide-in 0.6s ease",
    "&:hover": {
      transition: " 0.5s ease-in",
      backgroundColor: "rgba(127, 255, 212, 0.09)",
    },
  },

  Hidden: {
    // display: "none",
    animation: "$slide-out 0.6s ease",
    opacity: "0",
  },

  "@keyframes slide-in": {
    "0%": {
      opacity: "0",
      transform: "translateX(50px)",
    },
    "100%": {
      opacity: 1,
      transform: "translateX(0)",
    },
  },
  "@keyframes slide-out": {
    "0%": {
      opacity: 1,
      transform: "translateX(0)",
    },
    "100%": {
      opacity: 0,
      transform: "translateX(50px)",
    },
  },
}));

const ScrollToTop = ({ display, clicked }) => {
  const classes = useStyles();

  let cssClasses = [classes.ScrollToTop];

  if (!display) {
    cssClasses.push(classes.Hidden);
  }

  return (
    <div role="button" className={cssClasses.join(" ")} onClick={clicked}>
      ^
    </div>
  );
};

export default ScrollToTop;
