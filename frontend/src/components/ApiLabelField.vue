
<script lang="ts">
export default {
    props: {
        name: String,
        type: String,
        required: Boolean
    }
}
</script>

<template>
    <strong class="fieldLabel"> {{ name }} </strong> <p class="fieldType"> {{ type }} </p> <p  class="fieldRequired" v-if="required">*</p>
</template>

<style>
.fieldLabel{
    color: Black;
}

.fieldType{
    margin-left: 10px;
    color: Grey;
}

.fieldRequired{
    margin-left: 10px;
    color: red;
}
</style>