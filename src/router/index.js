import Vue from 'vue';
import Router from 'vue-router';
import AppLanding from '@/views/AppLanding.vue';  
import AppAnalysis from '@/views/AppAnalysis.vue'; 

Vue.use(Router);

export default new Router({
  mode: 'hash', 
  routes: [
    {
      path: '/', 
      name: 'Home',
      component: AppLanding,  
    },  
    {
      path: '/analisis', 
      name: 'Analysis',
      component: AppAnalysis,  
    },
  ],
});
