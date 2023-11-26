<script lang="ts">
import ApiLabelField from './ApiLabelField.vue';

import { HTTPSnippet } from 'httpsnippet';

export default {
    data() {
        return {
            prompt: '',
            database: '',
            model: '',
            gptVersion: '',
            init: false,
            validate: false,
            result: '',
            exception: '',
            waiting: false,
            tab: "javascript" as "c" | "clojure" | "csharp" | "go" | "http" | "java" | "javascript" | "kotlin" | "node" | "objc" | "ocaml" | "php" | "powershell" | "python" | "r" | "ruby" | "shell" | "swift",
            tabtext: ''
        };
    },
    components: {
        ApiLabelField
    },
    methods: {
        async submit() {
            var data = {
                "prompt": this.prompt === '' ? undefined : this.prompt,
                "database": this.database === '' ? undefined : this.database,
                "model": this.model === '' ? undefined : this.model,
                "gptVersion": this.gptVersion === '' ? undefined : this.gptVersion,
                "init": this.init === false ? undefined : this.init,
                "validate": this.validate === false ? undefined : this.validate,
            }
            var json = JSON.stringify(data)
            console.log(json);
            this.waiting = true;
            this.exception = "";
            try {
                const response = await fetch(window.location.origin + "/chat", {
                    method: 'POST',
                    headers: {
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    },
                    body: json,
                });
                response.json().then(data => {
                    this.result = JSON.stringify(data)
                    console.log(this.result);
                });
                
            }
            catch (ex) {
                if (ex instanceof TypeError) {
                    this.exception = ex.message
                }
                else {
                    console.log("Error" + ex)
                    this.exception = "" + ex;
                }
            }
            finally {
                this.waiting = false;
            }


        },
        ChangeExample(newValue: any) {
           
        }
    },
    computed : {
        getCode(){
            var data = {
                "prompt": this.prompt === '' ? undefined : this.prompt,
                "database": this.database === '' ? undefined : this.database,
                "model": this.model === '' ? undefined : this.model,
                "gptVersion": this.gptVersion === '' ? undefined : this.gptVersion,
                "init": this.init === false ? undefined : this.init,
                "validate": this.validate === false ? undefined : this.validate,
            }

            var json = JSON.stringify(data);
            const snippet = new HTTPSnippet({method: 'POST',postData: {
                mimeType:"",
                text : json
            }, url: window.location.origin + '/chat', httpVersion:"", cookies:[], headers:[], queryString:[], headersSize:-1, bodySize:-1} );

            const options = { indent: '\t' };
            const output = snippet.convert(this.tab, undefined, options);
            this.tabtext = output.toString();
            return this.tabtext;
        }
    }
}
</script>

<style>
.SectionText {
    color: green;
}

.LabelColor {
    font-weight: bold;
    color: #2656DA;
    background-color: #DFEDFE;
    border-radius: 10px;
    padding: 5px 10px 5px;
}

.ButtonTextColor {
    font-weight: bold;
    color: #DFEDFE;
}

.MainSheet {
    margin: 10px 0px 30px;
    padding: 20px;
}

.ToolBar {
    margin: 0px 0px 20px;
}

.Page {
    margin: 20px
}

.ResultField {
    margin: 0px 0px 30px;
}

.ErrorField {
    margin: 10px;
    color: red;

}

.ExampleField {
    margin: 0px 0px 20px;
}
</style>

<template>
    <div class="Page">
        <span class="text-h5 SectionText">API</span><br />

        <span class="text-h2">Ask a Question</span><br />

        <span class="text-body-1">This endpoint poses a question to the FH-Wedel ChatBot.</span><br />


        <v-sheet class="MainSheet" rounded border>
            <v-toolbar class="ToolBar">
                <template v-slot:prepend>
                    <span class="LabelColor">Post</span>
                </template>
                <v-divider class="ms-3" inset vertical></v-divider>
                <v-toolbar-title>/ chat</v-toolbar-title>

                <v-btn rounded="xs" color="#2656DA" class="text-none text-subtitle-1" variant="flat" @click="submit" :loading="waiting">
                    Send
                </v-btn>
            </v-toolbar>

            <v-expansion-panels>
                <v-expansion-panel title="Body">
                    <v-expansion-panel-text>
                        <v-textarea auto-grow v-model="prompt">
                            <template #label>
                                <ApiLabelField name="Prompt" type="string" :required='true' />
                            </template>
                        </v-textarea>
                        <v-combobox v-model="database" :items="['CustomLoader', 'PyPdfLoader']">
                            <template #label>
                                <ApiLabelField name="Database" type="enum<string>" />
                            </template>
                        </v-combobox>
                        <v-combobox v-model="model"
                            :items="['VecStore', 'TOOLS', 'CustomTool', 'SmartAgent', 'OpenAIAssistant']">
                            <template #label>
                                <ApiLabelField name="model" type="enum<string>" />
                            </template>
                        </v-combobox>
                        <v-combobox v-model="gptVersion"
                            :items="['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-4-32k-0613', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4', 'gpt-4-1106-preview']">
                            <template #label>
                                <ApiLabelField name="gptVersion" type="enum<string>" />
                            </template>
                        </v-combobox>
                        <v-checkbox v-model="init">
                            <template #label>
                                <ApiLabelField name="init" type="boolean" />
                            </template>
                        </v-checkbox>
                        <v-checkbox v-model="validate">
                            <template #label>
                                <ApiLabelField name="validate" type="boolean" />
                            </template>
                        </v-checkbox>
                    </v-expansion-panel-text>
                </v-expansion-panel>
            </v-expansion-panels>
        </v-sheet>

        <div class="ResultField" v-if="result != '' || exception != ''">
            <span class="text-h4">Antwort:</span>
            <v-card rounded="lg" v-if="result != ''">
                <highlightjs language='json' :code="result" />
            </v-card>
            <v-card rounded="lg" v-if="exception != ''">
                <span class="text-body-1 ErrorField">{{ exception }}</span>
            </v-card>
        </div>

        <div class="ExampleField">
            <span class="text-h4">Beispiel:</span>
            <v-card rounded="lg">
                <v-tabs bg-color="black" color="green" slider-color="green" v-model="tab"
                    @update:model-value="ChangeExample">
                    <v-tab value="javascript">javascript</v-tab>
                    <v-tab value="kotlin">clojure</v-tab>
                    <v-tab value="csharp">C#</v-tab>
                    <v-tab value="go">go</v-tab>
                    <v-tab value="shell">curl</v-tab>
                    <v-tab value="java">java</v-tab>
                    <v-tab value="ruby">ruby</v-tab>
                    <v-tab value="powershell">powershell</v-tab>
                    <v-tab value="python">python</v-tab>

                </v-tabs>
                <v-card-text style="padding: 0;">
                    <v-window v-model="tab">
                        <highlightjs :language='tab' :code="getCode" />
                    </v-window>
                </v-card-text>
            </v-card>
        </div>
    </div>
</template>
