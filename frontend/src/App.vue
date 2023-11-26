<script setup lang="ts">
import PlayGround from './components/PlayGround.vue';
</script>

<template>
  <!-- <PlayGround/> -->
  <v-layout class="rounded rounded-md">
    <v-app-bar title="FH Wedel ChatBot">
    </v-app-bar>

    <v-navigation-drawer>
      <v-list>
        <v-list-item title="Api">
          <v-list-item title="Chat">
          
          </v-list-item>
        </v-list-item>
       
      </v-list>
    </v-navigation-drawer>

    <v-main class="align-center justify-center" style="min-height: 300px;">
      <PlayGround/>
    </v-main>
  </v-layout>


</template>

<style scoped>

</style>
