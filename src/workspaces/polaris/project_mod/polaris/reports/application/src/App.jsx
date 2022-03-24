import React, { useReducer } from "react";
import { ThemeProvider } from "@material-ui/core/styles";
import { createTheme } from "@material-ui/core/styles";
import { createContext } from "react";
import {
  settingsInitialState,
  settingsReducer,
} from "./reducers/settingsReducer";
import themes from "./config/themes";
import Layout from "./Components/Layout";

export const StateContext = createContext();
export const DispatchContext = createContext();

const App = () => {
  const [state, dispatch] = useReducer(settingsReducer, settingsInitialState);
  const theme = (theme) =>
    createTheme({
      palette: themes[theme],
    });
  return (
    <DispatchContext.Provider value={dispatch}>
      <StateContext.Provider value={state}>
        <ThemeProvider theme={theme(state.currentTheme)}>
          <Layout />
        </ThemeProvider>
      </StateContext.Provider>
    </DispatchContext.Provider>
  );
};

export default App;
