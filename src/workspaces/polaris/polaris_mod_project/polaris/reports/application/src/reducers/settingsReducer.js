export const settingsInitialState = {
  currentTheme: localStorage.getItem("theme") === "dark" ? "dark" : "light",
};

export const settingsReducer = (state, { type, payload }) => {
  switch (type) {
    case "changeTheme": {
      localStorage.setItem("theme", payload);
      return { ...state, currentTheme: payload };
    }
    default:
      return state;
  }
};
