

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import OpenAI from 'openai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Initialize OpenAI clients
const clients = {};

// Hugging Face client
if (process.env.HF_TOKEN) {
    clients.huggingface = new OpenAI({
        baseURL: "https://api-inference.huggingface.co/v1",
        apiKey: process.env.HF_TOKEN,
    });
}

// OpenAI client
if (process.env.OPENAI_API_KEY) {
    clients.openai = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
    });
}

// Anthropic-compatible client (for other providers)
if (process.env.ANTHROPIC_API_KEY) {
    clients.anthropic = new OpenAI({
        baseURL: "https://api.anthropic.com/v1",
        apiKey: process.env.ANTHROPIC_API_KEY,
        defaultHeaders: {
            'anthropic-version': '2023-06-01'
        }
    });
}

// Available models for each provider
const MODELS = {
    huggingface: [
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill",
        "Qwen/Qwen2.5-Coder-32B-Instruct"
    ],
    openai: [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "o1-preview",
        "o1-mini"
    ],
    anthropic: [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229"
    ]
};

// Create MCP server
const server = new Server(
    {
        name: 'enhanced-ai-assistant',
        version: '2.0.0',
        description: 'Enhanced AI assistant with multiple model support'
    },
    {
        capabilities: {
            tools: {},
        },
    }
);

// Helper function to get available providers
function getAvailableProviders() {
    const available = [];
    if (clients.huggingface) available.push('huggingface');
    if (clients.openai) available.push('openai');
    if (clients.anthropic) available.push('anthropic');
    return available;
}

// Helper function to get default model for provider
function getDefaultModel(provider) {
    const defaults = {
        huggingface: "meta-llama/Llama-3.2-3B-Instruct",
        openai: "gpt-4o-mini",
        anthropic: "claude-3-5-haiku-20241022"
    };
    return defaults[provider];
}

// Register tool handlers , including generate_text and list_models by Mizanur Hasan Khan
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    if (name === 'generate_text') {
        const { prompt, provider, model, max_tokens, temperature } = args;

        if (!prompt) {
            return {
                content: [{
                    type: 'text',
                    text: 'Error: Prompt is required'
                }]
            };
        }

        const availableProviders = getAvailableProviders();

        if (availableProviders.length === 0) {
            return {
                content: [{
                    type: 'text',
                    text: `No AI providers configured. Please set one of the following environment variables:
- HF_TOKEN (for Hugging Face)
- OPENAI_API_KEY (for OpenAI)
- ANTHROPIC_API_KEY (for Anthropic)

Then restart the server.`
                }]
            };
        }

        // Use specified provider or default to first available
        const selectedProvider = provider && availableProviders.includes(provider)
            ? provider
            : availableProviders[0];

        const selectedModel = model || getDefaultModel(selectedProvider);
        const client = clients[selectedProvider];

        try {
            console.log(`Making request to ${selectedProvider} with model: ${selectedModel}`);

            const requestParams = {
                model: selectedModel,
                messages: [{
                    role: "user",
                    content: prompt,
                }],
                max_tokens: max_tokens || 1024,
                temperature: temperature || 0.7
            };

            const response = await client.chat.completions.create(requestParams);
            const generatedText = response.choices[0]?.message?.content || 'No response generated';

            return {
                content: [{
                    type: 'text',
                    text: generatedText
                }]
            };

        } catch (error) {
            console.error(`${selectedProvider} API Error:`, error);

            let errorMessage = 'Failed to generate text';
            if (error.response?.status === 401) {
                errorMessage = `Authentication failed for ${selectedProvider}. Please check your API key.`;
            } else if (error.response?.status === 429) {
                errorMessage = `Rate limit exceeded for ${selectedProvider}. Please try again later.`;
            } else if (error.message) {
                errorMessage = error.message;
            }

            return {
                content: [{
                    type: 'text',
                    text: `Error: ${errorMessage}`
                }]
            };
        }
    }

    if (name === 'list_models') {
        const availableProviders = getAvailableProviders();

        if (availableProviders.length === 0) {
            return {
                content: [{
                    type: 'text',
                    text: 'No AI providers configured.'
                }]
            };
        }

        let modelList = 'Available AI Models:\n\n';

        for (const provider of availableProviders) {
            modelList += `**${provider.toUpperCase()}:**\n`;
            for (const model of MODELS[provider]) {
                const isDefault = model === getDefaultModel(provider);
                modelList += `- ${model}${isDefault ? ' (default)' : ''}\n`;
            }
            modelList += '\n';
        }

        return {
            content: [{
                type: 'text',
                text: modelList
            }]
        };
    }

    if (name === 'chat_conversation') {
        const { messages, provider, model, max_tokens, temperature } = args;

        if (!messages || !Array.isArray(messages) || messages.length === 0) {
            return {
                content: [{
                    type: 'text',
                    text: 'Error: Messages array is required'
                }]
            };
        }

        const availableProviders = getAvailableProviders();

        if (availableProviders.length === 0) {
            return {
                content: [{
                    type: 'text',
                    text: 'No AI providers configured.'
                }]
            };
        }

        const selectedProvider = provider && availableProviders.includes(provider)
            ? provider
            : availableProviders[0];

        const selectedModel = model || getDefaultModel(selectedProvider);
        const client = clients[selectedProvider];

        try {
            console.log(`Chat request to ${selectedProvider} with model: ${selectedModel}`);

            const requestParams = {
                model: selectedModel,
                messages: messages,
                max_tokens: max_tokens || 1024,
                temperature: temperature || 0.7
            };

            const response = await client.chat.completions.create(requestParams);
            const generatedText = response.choices[0]?.message?.content || 'No response generated';

            return {
                content: [{
                    type: 'text',
                    text: generatedText
                }]
            };

        } catch (error) {
            console.error(`${selectedProvider} Chat API Error:`, error);

            let errorMessage = 'Failed to process chat';
            if (error.response?.status === 401) {
                errorMessage = `Authentication failed for ${selectedProvider}. Please check your API key.`;
            } else if (error.response?.status === 429) {
                errorMessage = `Rate limit exceeded for ${selectedProvider}. Please try again later.`;
            } else if (error.message) {
                errorMessage = error.message;
            }

            return {
                content: [{
                    type: 'text',
                    text: `Error: ${errorMessage}`
                }]
            };
        }
    }

    return {
        content: [{
            type: 'text',
            text: `Unknown tool: ${name}`
        }]
    };
});

// Register tool definitions
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: 'generate_text',
                description: 'Generate text using AI models from various providers (Hugging Face, OpenAI, Anthropic)',
                inputSchema: {
                    type: 'object',
                    properties: {
                        prompt: {
                            type: 'string',
                            description: 'The prompt to send to the AI model'
                        },
                        provider: {
                            type: 'string',
                            description: 'AI provider to use (huggingface, openai, anthropic)',
                            enum: ['huggingface', 'openai', 'anthropic']
                        },
                        model: {
                            type: 'string',
                            description: 'Specific model to use (optional, will use provider default)'
                        },
                        max_tokens: {
                            type: 'number',
                            description: 'Maximum tokens to generate (default: 1024)',
                            default: 1024
                        },
                        temperature: {
                            type: 'number',
                            description: 'Temperature for text generation (0.0-2.0, default: 0.7)',
                            default: 0.7,
                            minimum: 0.0,
                            maximum: 2.0
                        }
                    },
                    required: ['prompt']
                }
            },
            {
                name: 'list_models',
                description: 'List all available AI models from configured providers',
                inputSchema: {
                    type: 'object',
                    properties: {},
                    additionalProperties: false
                }
            },
            {
                name: 'chat_conversation',
                description: 'Have a multi-turn conversation with AI models',
                inputSchema: {
                    type: 'object',
                    properties: {
                        messages: {
                            type: 'array',
                            description: 'Array of chat messages with role and content',
                            items: {
                                type: 'object',
                                properties: {
                                    role: {
                                        type: 'string',
                                        enum: ['system', 'user', 'assistant']
                                    },
                                    content: {
                                        type: 'string'
                                    }
                                },
                                required: ['role', 'content']
                            }
                        },
                        provider: {
                            type: 'string',
                            description: 'AI provider to use (huggingface, openai, anthropic)',
                            enum: ['huggingface', 'openai', 'anthropic']
                        },
                        model: {
                            type: 'string',
                            description: 'Specific model to use (optional, will use provider default)'
                        },
                        max_tokens: {
                            type: 'number',
                            description: 'Maximum tokens to generate (default: 1024)',
                            default: 1024
                        },
                        temperature: {
                            type: 'number',
                            description: 'Temperature for text generation (0.0-2.0, default: 0.7)',
                            default: 0.7,
                            minimum: 0.0,
                            maximum: 2.0
                        }
                    },
                    required: ['messages']
                }
            }
        ]
    };
});

// Error handling
process.on('SIGINT', async () => {
    console.log('\nShutting down MCP server...');
    await server.close();
    process.exit(0);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Start the server
async function startServer() {
    try {
        const transport = new StdioServerTransport();
        await server.connect(transport);
        console.log('Enhanced MCP AI Assistant Server running...');
        console.log('Available providers:', getAvailableProviders());
    } catch (error) {
        console.error('Failed to start server:', error);
        process.exit(1);
    }
}

startServer();