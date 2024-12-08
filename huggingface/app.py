import gradio as gr
from pathlib import Path
import shutil
from gradio_pdf import PDF
from functools import partial

from caplang.apps.rag.langchainapps import ChatBot

class GradioChat:
    def __init__(self, chatbot: ChatBot):
        self.chatbot = chatbot
    
    def upload_file(self, filepath, to_kb=True):
        print(filepath)
        shutil.copy(filepath, "kb")
        file_name = Path(filepath).name
        new_file_path = Path("kb") / file_name
        if to_kb:
            # update new embedding kb
            self.chatbot.update_kb_with_file(new_file_path)
        else:
            # create new embedding kb
            self.chatbot.create_kb_with_file(new_file_path)
    
    def pdf_chat(self, question, document):
        self.upload_file(document)
        return self.chatbot.query_by_doc(question, document, session_id="test")
            
    def predict(self, message, history, use_kb=True):
        history_langchain_format = []
        for human, ai in history:
            history_langchain_format.append(message)
            history_langchain_format.append(ai)
        history_langchain_format.append(message)
        if use_kb:
            ans = self.chatbot.query(message, session_id="test")
        else:
            ans = self.chatbot.query_by_doc(message, session_id="test")
        return ans

    def update_message(self, request: gr.Request) -> gr.Blocks:
        return f"Welcome, {request.username}"
        
    def launch(self, **kwargs):
        css = """
        #chatbot {
            flex-grow: 1 !important;
            overflow: auto !important;
        }
        #col { height: calc(100vh - 112px - 16px - 150px) !important; }
        #docs { height: calc(100vh - 112px - 16px - 150px) !important; }
        #pdf { height: 100% !important; }
        .pdf-canvas, canvas { height: cal(100% - 40px) !important; }
        """
        with gr.Blocks(fill_height=True, css=css, theme=gr.themes.Soft()) as demo:
            gr.HTML('<img src="https://www.ntu-cap.org/wp-content/uploads/2020/12/cap_banner_top_padding.png" alt="logo" width="400" height="100">')
            gr.Markdown("# DCGPT: An open-source project for data center large language model")
            m = gr.Markdown()
            demo.load(self.update_message, None, m)
            with gr.Tab("Chat with DCGPT"):
                with gr.Column(elem_id="col"):
                    gr.ChatInterface(
                        fn=self.predict,
                        chatbot=gr.Chatbot(elem_id="chatbot",
                                    render=False),
                    )
            with gr.Tab("Chat with Docs"):
                with gr.Row(equal_height=True, elem_id="docs"):
                    with gr.Column(scale=2, min_width=300, show_progress=True):
                        pdf = PDF(label="Document", min_width=300, height=400, interactive=True, elem_id="pdf")
                        pdf.upload(partial(self.upload_file, to_kb=False), pdf)
                    with gr.Column(scale=1, min_width=300):
                        gr.ChatInterface(partial(self.predict, use_kb=False),fill_height=True)
            
        demo.launch(auth=self.auth, **kwargs)
        
        
if __name__ == "__main__":
    chatbot = ChatBot("kb/chroma_db", model="dcgpt")
    gr_chat = GradioChat(chatbot)
    gr_chat.launch(server_name="0.0.0.0", share=True)