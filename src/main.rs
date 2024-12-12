mod model;
use actix_web::{self, Responder};
use actix_web::{
    body::MessageBody,
    dev::{ServiceRequest, ServiceResponse},
    middleware::{from_fn, Next},
    Error,
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tokio::sync::RwLock;

#[derive(Deserialize)]
struct Prompt {
    text: String,
}

#[derive(Serialize)]
struct Embeddings {
    ys: Vec<Vec<Vec<f32>>>,
}

struct AppModel {
    // bert: Mutex<model::Bert>,
    bert: RwLock<model::Bert>,
}

#[actix_web::post("/predict")]
async fn predict(
    prompt: actix_web::web::Json<Prompt>,
    model: actix_web::web::Data<AppModel>,
) -> impl Responder {
    let model = model.bert.read().await;

    println!("Predicting: {:?}", prompt.text);

    let ys = model
        .predict(prompt.text.clone())
        .unwrap()
        .to_vec3()
        .unwrap();

    actix_web::web::Json(Embeddings { ys })
}

async fn middleware_time_elapsed(
    req: ServiceRequest,
    next: Next<impl MessageBody>,
) -> Result<ServiceResponse<impl MessageBody>, Error> {
    let payload_size = req
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);

    println!("Payload size: {:?}", payload_size);

    let now = std::time::Instant::now();

    let res = next.call(req).await;

    println!("Request took: {:?}", now.elapsed());

    res
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = actix_web::web::Data::new(AppModel {
        bert: RwLock::new(model::Bert::new(None, None, false, false).unwrap()),
    });

    println!("Current PID: {:?}", std::process::id());

    actix_web::HttpServer::new(move || {
        actix_web::App::new()
            // TODO: try without lock
            // .app_data(actix_web::web::Data::new(AppModel {
            //     bert: RwLock::new(model::Bert::new(None, None, false, false).unwrap()),
            // }))
            .app_data(model.clone())
            .wrap(from_fn(middleware_time_elapsed))
            .service(predict)
    })
    .workers(8)
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

// curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello"}' http://localhost:8080/predict
//
// Get memory usage
// ps -p  58447 -o %cpu,rss= | awk '{print $1, $2/1024 " MB"}'
