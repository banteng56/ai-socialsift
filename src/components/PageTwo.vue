<script setup>
import { onMounted } from 'vue';
import axios from 'axios';
import cytoscape from 'cytoscape';
import coseBilkent from 'cytoscape-cose-bilkent';
import Heatmap from './heatmat.vue'; 

cytoscape.use(coseBilkent);

const loadGraphData = async () => {
  try {
    const cluster1 = await axios.get('/graph_cluster_0.json');
    const cluster2 = await axios.get('/graph_cluster_1.json');
    const cluster3 = await axios.get('/graph_cluster_2.json');

    const elements = [
      ...cluster1.data.elements.map((el) => ({ ...el, data: { ...el.data, cluster: 0 } })),
      ...cluster2.data.elements.map((el) => ({ ...el, data: { ...el.data, cluster: 1 } })),
      ...cluster3.data.elements.map((el) => ({ ...el, data: { ...el.data, cluster: 2 } })),
    ];

    return elements;
  } catch (error) {
    console.error('Error loading graph data:', error);
    return [];
  }
};

onMounted(async () => {
  const elements = await loadGraphData();

  const sourceNodes = new Set();
  const targetNodes = new Set();

  elements.forEach((el) => {
    if (el.data.source) sourceNodes.add(el.data.source);
    if (el.data.target) targetNodes.add(el.data.target);
  });

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': (ele) => {
            const id = ele.data('id');
            const cluster = ele.data('cluster');

            if (targetNodes.has(id)) return '#007BFF';

            if (sourceNodes.has(id)) {
              return cluster === 0 ? '#FF5733' : cluster === 1 ? '#28A745' : '#FFC107';
            }

            return '#CCCCCC';
          },
          width: (ele) => (targetNodes.has(ele.data('id')) ? '15px' : '25px'),
          height: (ele) => (targetNodes.has(ele.data('id')) ? '15px' : '25px'),
          label: 'data(label)',
          'text-valign': 'top',
          'text-halign': 'center',
          'text-margin-y': '-10px',
          color: '#000',
          'font-size': '10px',
          opacity: 1,
          'text-opacity': (ele) => (targetNodes.has(ele.data('id')) ? 0 : 1),
        },
      },
      {
        selector: 'edge',
        style: {
          width: 2,
          'line-color': '#ccc',
          'target-arrow-color': '#ccc',
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          label: 'data(label)',
          'font-size': 8,
          color: '#666',
          'text-opacity': 0,
          'text-rotation': 'autorotate',
          opacity: 1,
        },
      },
      {
        selector: '.faded',
        style: {
          opacity: 0.1,
        },
      },
      {
        selector: '.highlighted',
        style: {
          opacity: 1,
          'line-color': '#FF5733',
          'target-arrow-color': '#FF5733',
        },
      },
      {
        selector: '.blurred',
        style: {
          filter: 'blur(5px)',
          transition: 'filter 0.2s ease',
        },
      },
    ],
    layout: {
      name: 'cose-bilkent',
      idealEdgeLength: 150,
      nodeRepulsion: 8000,
      gravity: 0.2,
    },
  });

  cy.on('tap', 'node', (evt) => {
    const node = evt.target;

    cy.elements().removeClass('faded');
    cy.elements().removeClass('highlighted');
    cy.elements().removeClass('blurred');

    cy.nodes().forEach((n) => {
      if (n.data('background-color') === '#007BFF') {
        n.style('text-opacity', 0);
      }
    });

    node.style('text-opacity', 1);

    node.addClass('highlighted');
    const connectedEdges = node.connectedEdges();
    connectedEdges.addClass('highlighted');
    connectedEdges.targets().addClass('highlighted');
    connectedEdges.sources().addClass('highlighted');

    const connectedBiruNodes = connectedEdges.targets().filter((ele) => targetNodes.has(ele.data('id')));
    connectedBiruNodes.forEach((biruNode) => {
      biruNode.style('text-opacity', 1);
    });

    cy.nodes().not(node).addClass('blurred');
  });

  cy.on('tap', (evt) => {
    if (evt.target === cy) {
      cy.elements().removeClass('faded');
      cy.elements().removeClass('highlighted');
      cy.elements().removeClass('blurred');
      cy.nodes().style('text-opacity', (ele) => (targetNodes.has(ele.data('id')) ? 0 : 1));
    }
  });

  cy.on('zoom', () => {
    const zoomLevel = cy.zoom();
    cy.edges().forEach((edge) => {
      edge.style('text-opacity', zoomLevel > 1.5 ? 1 : 0);
    });
  });

  cy.on('drag', 'node', (evt) => {
    const node = evt.target;
    const position = node.position();

    const connectedEdges = node.connectedEdges();
    const connectedNodes = connectedEdges.targets().add(connectedEdges.sources());

    const dx = position.x - node.scratch('_prevPosX');
    const dy = position.y - node.scratch('_prevPosY');

    connectedNodes.forEach((connectedNode) => {
      const currentPos = connectedNode.position();
      connectedNode.position({
        x: currentPos.x + dx,
        y: currentPos.y + dy,
      });
    });

    node.scratch('_prevPosX', position.x);
    node.scratch('_prevPosY', position.y);
  });

  cy.nodes().forEach((node) => {
    const position = node.position();
    node.scratch('_prevPosX', position.x);
    node.scratch('_prevPosY', position.y);
  });
});
</script>

<template>
  <v-container>
    <h1>Graph Entities</h1>
    <div id="cy" style="height: 600px; border: 1px solid #ccc;"></div>
    <h1>Heatmap Entities</h1>
    <Heatmap /> 
  </v-container>
</template>

<style scoped>
#cy {
  border-radius: 8px;
}
</style>
