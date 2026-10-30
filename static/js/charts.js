const frequencySelect = document.querySelector("#frequency");
const horizonSelect = document.querySelector("#horizon");

if (frequencySelect && horizonSelect) {
	function updateHorizonOptions() {
		const selectedFrequency = frequencySelect.value;
		const availableOptions = Array.from(horizonSelect.options).filter((option) => {
			const isAvailable = option.dataset.frequency === selectedFrequency;
			option.hidden = !isAvailable;
			option.disabled = !isAvailable;
			return isAvailable;
		});

		if (!availableOptions.some((option) => option.value === horizonSelect.value)) {
			horizonSelect.value = availableOptions[0].value;
		}
	}

	frequencySelect.addEventListener("change", updateHorizonOptions);
	updateHorizonOptions();
}

const chartDataElement = document.querySelector("#forecast-chart-data");
const chartCanvas = document.querySelector("#forecast-chart");

if (chartDataElement && chartCanvas && window.Chart) {
	const chartData = JSON.parse(chartDataElement.textContent);

	new Chart(chartCanvas, {
		type: "line",
		data: {
			labels: chartData.labels,
			datasets: [
				{
					label: "Historical sales",
					data: chartData.historical,
					borderColor: "#2875d2",
					backgroundColor: "rgba(40, 117, 210, 0.12)",
					pointRadius: 0,
					borderWidth: 2,
					tension: 0.2,
					spanGaps: false,
				},
				{
					label: "Forecast sales",
					data: chartData.forecast,
					borderColor: "#e68a32",
					backgroundColor: "rgba(230, 138, 50, 0.12)",
					pointRadius: 3,
					borderWidth: 2,
					borderDash: [6, 4],
					tension: 0.2,
					spanGaps: false,
				},
			],
		},
		options: {
			responsive: true,
			maintainAspectRatio: false,
			interaction: { intersect: false, mode: "index" },
			scales: {
				y: {
					ticks: {
						callback(value) {
							return `₹${Number(value).toLocaleString("en-IN")}`;
						},
					},
				},
			},
		},
	});
}