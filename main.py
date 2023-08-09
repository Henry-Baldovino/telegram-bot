from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Final
from aiogram import Bot
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
import openai
import telegram
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

import os
from sentence_transformers import SentenceTransformer
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

import requests
from model import generation_response

plt.style.use("ggplot")

TOKEN = "6142192074:AAFaBz8QkA1DHT8-VWTnNxHJS5JGbaZCG2A"
BOT_USERNAME: Final = "@VinayKbot"

# Add Commands


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hello! Thanks for chatting with me! I am Viany Kayal and project manager of 'Operation Nova'.😎"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "I am Vinay Kayal, Please type something so I can respond!"
    )


async def custom_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("This is custom command!")


# Generate a response using ChatGPT

# def generate_response(text: str) -> str:
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an intelligent Assistant.",
#             },
#             {"role": "user", "content": text},
#         ],
#         temperature=0.8,
#         max_tokens=50,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#     )
#     return response.choices[0].message.content


# Responses

def handle_response(text: str) -> str:

    processed: str = text.lower()

    if "hello" in processed:
        return "Hello, How are you?"

    if "how are you" in processed:
        return "I am good!"

    if "thanks" in processed:
        return "You are Welcome!🎇"

    if "i am good" in processed:
        return "Okay"

    if "you are welcome" in processed:
        return "Ask me anything, I will answer the question."

    return generation_response(processed)


sia = SentimentIntensityAnalyzer()


def sentiment_analysis(text: str):

    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged)
    entities.pprint()
    emotion = sia.polarity_scores(text)
    return emotion


# if update.edited_message is not None:
#     print(f"edited message :{update.edited_message.text}")
#     text: str = update.edited_message.text
#     return
# elif update.edited_channel_post is not None:
#     print(f"edited channel post: {update.edited_channel_post.text}")
#     return


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:

        message_type: str = update.message.chat.type
        message_id = update.message.message_id
        chat_id: str = update.message.chat.id
        print(f"message_id: {message_id}, chat_id: {chat_id}")

        if message_type == "group" or message_type == "private":
            text: str = update.message.text
            text = text.lower()
            print(f'User({update.message.chat.id}) in {message_type}:"{text}"')
            sentiment = sentiment_analysis(text)
            print(sentiment["compound"])
            if sentiment["compound"] < 0:
                await context.bot.deleteMessage(message_id=message_id, chat_id=chat_id)

            if "vinay" in text and message_type == "group":
                print(text)
                response = handle_response(text)
                print("Bot:", response)
                await update.message.reply_text(response)

            if message_type == "private":
                print(text)
                response = handle_response(text)
                print("Bot:", response)
                await update.message.reply_text(response)

            return

    except:
        text: str = update.channel_post.text
        # message_id = update.channel_post.message_id
        # chat_id: str = update.message.chat.id
        text = text.lower()
        # sentiment = sentiment_analysis(text)
        # print(sentiment["compound"])
        # if sentiment["compound"] < 0:
        #     await context.bot.deleteMessage(message_id=message_id, chat_id=chat_id)

        if "vinay" in text:
            response: str = handle_response(text)
            print("In channel, Bot:", response)
            await update.channel_post.reply_text(response)


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"Update {update} caused error {context.error}")


if __name__ == "__main__":
    print("Starting bot...")
    app = Application.builder().token(TOKEN).build()

    # Commands

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("custom", custom_command))

    # Messages

    app.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Errors

    app.add_error_handler(error)

    # Polls the bot

    print("Polling...")
    app.run_polling(poll_interval=3)
