import { makeStyles } from "@material-ui/core";
import React, { memo, useState } from "react";
import { useEffect } from "react";
import HomeView from "../views/HomeView";
import ScrollToTop from "./common/ScrollToTop";
import Navbar from "./common/Navbar";
import { StateContext } from "../App";
import { useContext } from "react";

const useStyles = makeStyles((theme) => ({
  content: {
    padding: "13rem 5% 13rem 5%",
    backgroundColor: theme.palette.background.default,
    color: theme.palette.text.primary,
    transition: "0.5s ease-in",
  },
}));

const Layout = memo(() => {
  const classes = useStyles();
  const state = useContext(StateContext);
  const { currentTheme } = state;
  const [scrollState, setScrollState] = useState({
    navbarStacked: true,
    displayScrollToTop: false,
    navbarHidden: false,
  });

  useEffect(() => {
    window.addEventListener("scroll", scrollHandler);
    return () => {
      window.removeEventListener("scroll", scrollHandler);
    };
  }, []);

  const scrollHandler = () => {
    if (window.scrollY > 35) {
      setScrollState((state) => ({ ...state, navbarStacked: true }));
    } else {
      setScrollState((state) => ({ ...state, navbarStacked: false }));
    }

    if (window.scrollY >= 225) {
      setScrollState((state) => ({
        ...state,
        navbarHidden: true,
        displayScrollToTop: true,
      }));
    } else {
      setScrollState((state) => ({
        ...state,
        navbarHidden: false,
        displayScrollToTop: false,
      }));
    }
  };
  const scrollToTopHandler = () => {
    window.scrollTo({
      top: 0,
      left: 0,
      behavior: "smooth",
    });
  };
  return (
    <>
      <Navbar
        stacked={scrollState.navbarStacked}
        hidden={scrollState.navbarHidden}
      />
      <main className={`${classes.content} ${currentTheme}`}>
        <HomeView />
      </main>
      <ScrollToTop
        display={scrollState.displayScrollToTop}
        clicked={scrollToTopHandler}
      />
    </>
  );
});

export default Layout;
