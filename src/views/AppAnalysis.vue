<template>
    <v-app>
      <v-main>
        <page-two />
      </v-main>
    </v-app>
  </template>
  
  <script>
  import PageTwo from '../components/PageTwo';
  
  export default {
    name: 'AppAnalysis',
    components: {
      PageTwo,
    },
  };
  </script>
  