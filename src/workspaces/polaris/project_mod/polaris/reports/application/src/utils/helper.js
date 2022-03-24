export function download(content, fileName, contentType) {
  var a = document.createElement("a");
  var file = new Blob([content], { type: contentType });
  a.href = URL.createObjectURL(file);
  a.download = fileName;
  a.click();
}

export function getCommon(arr1, arr2) {
  arr1.sort();
  arr2.sort();
  var common = [];
  var i = 0,
    j = 0;
  while (i < arr1.length && j < arr2.length) {
    if (arr1[i] === arr2[j]) {
      common.push(arr1[i]);
      i++;
      j++;
    } else if (arr1[i] < arr2[j]) {
      i++;
    } else {
      j++;
    }
  }

  return common;
}

export const toShortFormat = (date) => {
  let monthNames = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
  ];

  let day = date.getDate();

  let monthIndex = date.getMonth();
  let monthName = monthNames[monthIndex];

  let year = date.getFullYear();

  return `${monthName} ${day}, ${year}`;
};

export const getRandomColor = () => {
  const r = Math.floor(Math.random() * 200);
  const g = Math.floor(Math.random() * 200);
  const b = Math.floor(Math.random() * 200);
  const color = "rgb(" + r + ", " + g + ", " + b + ")";
  return color;
};
