<template>
    <div ref="treemapContainer" class="treemap-container"></div>
  </template>
  
  <script setup>
  import { ref, onMounted } from "vue";
  import * as d3 from "d3";
  
  const treemapContainer = ref(null);
  
  onMounted(async () => {
    const response = await fetch("/recap_data.json");
    const rawData = await response.json();
  
    const data = {
      name: "root",
      children: Object.entries(rawData).map(([key, value]) => ({
        name: key,
        value: value["Total Count"],
        classification: Object.keys(value["Classification"])[0], 
        sentiment: Object.keys(value["Sentiment"])[0], 
      })),
    };
  
    const width = 800;
    const height = 400;
  
    const svg = d3
      .select(treemapContainer.value)
      .append("svg")
      .attr("width", width)
      .attr("height", height)
      .style("font-family", "sans-serif");
  
    const root = d3
      .hierarchy(data)
      .sum((d) => d.value)
      .sort((a, b) => b.value - a.value);
  
    d3.treemap().size([width, height]).padding(2)(root);
  
    const nodes = svg
      .selectAll("g")
      .data(root.leaves())
      .enter()
      .append("g")
      .attr("transform", (d) => `translate(${d.x0},${d.y0})`);
  
    nodes
      .append("rect")
      .attr("width", (d) => d.x1 - d.x0)
      .attr("height", (d) => d.y1 - d.y0)
      .attr("fill", (d) => d3.interpolateBlues(d.value / 20)) 
      .style("stroke", "#fff")
      .on("click", (event, d) => {
        alert(`Name: ${d.data.name}\nClassification: ${d.data.classification}\nSentiment: ${d.data.sentiment}`);
      });
  
    nodes
      .append("text")
      .attr("x", 5)
      .attr("y", 20)
      .text((d) => d.data.name)
      .attr("fill", "#000")
      .style("font-size", "12px")
      .style("pointer-events", "none");
  });
  </script>
  
  <style scoped>
  .treemap-container {
    width: 100%;
    overflow: hidden;
  }
  </style>
  